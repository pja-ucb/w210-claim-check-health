from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


DROP_COLS = {
    "DESYNPUF_ID",
    "CLM_ID",
    "CLM_PMT_AMT",
    "NCH_PRMRY_PYR_CLM_PD_AMT",
    "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",
    "NCH_BENE_PTB_DDCTBL_AMT",
    "NCH_BENE_PTB_COINSRNC_AMT",
    "Zero_Amount_Signal",
    "Claim_Adjustment_Signal",
    "High_HCPCS_Count_Signal",
    "High_Diagnosis_Count_Signal",
    "Weak_Primary_Diagnosis_Signal",
    "High_Intensity_Weak_Dx_Signal",
    "Any_High_Intensity_HCPCS_Signal",
    "Missing_Primary_Diagnosis_Signal",
    "Missing_Provider_Number_Signal",
    "High_Risk",
}


@dataclass
class ModelArtifacts:
    model: Any
    encoder: Optional[Any]
    scaler: Any
    feature_cols: list[str]
    threshold: float


class LocalModelClient:
    def __init__(
        self,
        outpatient_dir: str,
        inpatient_dir: Optional[str] = None,
    ) -> None:
        self.outpatient = self._load_dir(outpatient_dir)
        self.inpatient = self._load_dir(inpatient_dir) if inpatient_dir else None

    def score(self, fields: Dict[str, Any], claim_type: str) -> tuple[float, float, Dict[str, Any]]:
        artifacts = self.outpatient if claim_type != "inpatient" else self.inpatient
        if artifacts is None:
            return 0.0, 1.0, {"error": "Model artifacts missing"}
        X, debug = self._prepare_features(fields, artifacts)
        prob = float(artifacts.model.predict(X, verbose=0).ravel()[0])
        debug.update(
            {
                "input_min": float(np.min(X)),
                "input_max": float(np.max(X)),
                "input_mean": float(np.mean(X)),
            }
        )
        return prob, artifacts.threshold, debug

    def _load_dir(self, model_dir: Optional[str]) -> Optional[ModelArtifacts]:
        if not model_dir:
            return None

        model_path = os.path.join(model_dir, "outpatient_nn.keras")
        encoder_path = os.path.join(model_dir, "outpatient_encoder.pkl")
        scaler_path = os.path.join(model_dir, "outpatient_scaler.pkl")
        feature_cols_path = os.path.join(model_dir, "outpatient_feature_cols.pkl")
        threshold_path = os.path.join(model_dir, "outpatient_thresholds.json")

        if not os.path.exists(model_path):
            return None

        model = load_model(model_path)
        encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(feature_cols_path)

        threshold = 0.5
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                data = json.load(f)
            threshold = data.get("best_threshold") or data.get("default_threshold") or threshold

        return ModelArtifacts(
            model=model,
            encoder=encoder,
            scaler=scaler,
            feature_cols=feature_cols,
            threshold=float(threshold),
        )

    def _prepare_features(self, fields: Dict[str, Any], artifacts: ModelArtifacts) -> tuple[np.ndarray, Dict[str, Any]]:
        row = {k: fields.get(k) for k in artifacts.feature_cols}

        if "CLM_FROM_DT" in fields and ("claim_month" in artifacts.feature_cols or "claim_year" in artifacts.feature_cols):
            dt = pd.to_datetime(str(fields.get("CLM_FROM_DT")), format="%Y%m%d", errors="coerce")
            if "claim_month" in artifacts.feature_cols and pd.notna(dt):
                row["claim_month"] = int(dt.month)
            if "claim_year" in artifacts.feature_cols and pd.notna(dt):
                row["claim_year"] = int(dt.year)

        df = pd.DataFrame([row])
        df = df.drop(columns=[c for c in df.columns if c in DROP_COLS], errors="ignore")

        cat_cols: list[str] = []
        if artifacts.encoder is not None and hasattr(artifacts.encoder, "feature_names_in_"):
            cat_cols = list(artifacts.encoder.feature_names_in_)
        else:
            cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        unknown_count = 0
        if cat_cols and artifacts.encoder is not None:
            filled = df[cat_cols].copy()
            # Normalize empty strings and whitespace to match training fillna("__NA__")
            for col in cat_cols:
                filled[col] = filled[col].apply(
                    lambda v: "__NA__" if v is None or str(v).strip() == "" else v
                )
            for i, col in enumerate(cat_cols):
                try:
                    cats = artifacts.encoder.categories_[i]
                    sample = next((c for c in cats if c is not None), None)
                    if isinstance(sample, (int, float, np.integer, np.floating)):
                        filled[col] = pd.to_numeric(filled[col], errors="coerce").fillna(-1)
                    elif isinstance(sample, (bytes, bytearray)):
                        filled[col] = filled[col].astype(str).apply(lambda v: v.encode())
                    else:
                        filled[col] = filled[col].fillna("__NA__").astype(str)

                    allowed = set(cats)
                    if filled.iloc[0][col] not in allowed:
                        unknown_count += 1
                except Exception:
                    filled[col] = filled[col].fillna("__NA__").astype(str)
            df[cat_cols] = artifacts.encoder.transform(filled)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

        X_scaled = artifacts.scaler.transform(df[artifacts.feature_cols])
        missing_cols = [c for c in artifacts.feature_cols if c not in fields]
        debug = {
            "total_features": len(artifacts.feature_cols),
            "missing_features": len(missing_cols),
            "missing_features_sample": missing_cols[:10],
            "unknown_category_features": unknown_count,
        }
        return X_scaled, debug
