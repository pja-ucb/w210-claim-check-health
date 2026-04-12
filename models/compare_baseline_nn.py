"""
Compare neural network risk model with a baseline (logistic regression).
Uses same features and same train/val/test split. Outputs metrics table and
per-claim predictions so you can compare what each model predicted for a claim.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Optional: PyTorch for neural network. Baseline (logistic regression) always runs.
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    pass

# Paths
DATA_DIR = Path(__file__).resolve().parent
# Use NN file if present, else fallback to nn_training_data.csv
NN_CSV = DATA_DIR / "Neural Net" / "outpatient_claims_nn.csv"
FALLBACK_CSV = DATA_DIR / "nn_training_data.csv"
OUTPUT_DIR = DATA_DIR
PREDICTIONS_CSV = OUTPUT_DIR / "baseline_vs_nn_predictions.csv"
COMPARISON_HTML = OUTPUT_DIR / "baseline_vs_nn_comparison.html"

# Binary signals to use as features (from dataset)
FEATURE_COLUMNS = [
    "diagnosis_is_other",
    "has_primary_diagnosis",
    "has_primary_procedure",
    "Missing_Primary_Diagnosis_Signal",
    "High_HCPCS_Count_Signal",
    "High_Diagnosis_Count_Signal",
    "Missing_Provider_Number_Signal",
    "Weak_Primary_Diagnosis_Signal",
    "Any_High_Intensity_HCPCS_Signal",
    "High_Intensity_Weak_Dx_Signal",
]
TARGET = "Final_Score"

# Extra columns: SEGMENT (one-hot), CLM_FROM_DT/CLM_THRU_DT -> Claim_Duration_Days, Claim_Month, Claim_Year
DATE_FROM_COL = "CLM_FROM_DT"
DATE_THRU_COL = "CLM_THRU_DT"
SEGMENT_COL = "SEGMENT"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of full; train = 1 - TEST - VAL
EPOCHS = 30
BATCH_SIZE = 2048
LR = 1e-3


def _parse_date(ser):
    """Parse YYYYMMDD (int or str) to datetime; preserves index, invalid -> NaT."""
    ser = pd.to_numeric(ser, errors="coerce")
    valid = ser.notna()
    if not valid.any():
        return pd.Series(pd.NaT, index=ser.index)
    conv = ser.loc[valid].astype(int).astype(str).str.zfill(8)
    dt = pd.to_datetime(conv, format="%Y%m%d", errors="coerce")
    out = pd.Series(pd.NaT, index=ser.index, dtype="datetime64[ns]")
    out.loc[valid] = dt.values
    return out


def load_data():
    if NN_CSV.exists():
        df = pd.read_csv(NN_CSV, low_memory=False)
    else:
        df = pd.read_csv(FALLBACK_CSV, low_memory=False)
    feats = [c for c in FEATURE_COLUMNS if c in df.columns]
    if TARGET not in df.columns or len(feats) == 0:
        raise RuntimeError("Missing feature or target columns")

    # Subset and filter binary + target
    use_cols = feats + [TARGET]
    if SEGMENT_COL in df.columns:
        use_cols = [SEGMENT_COL] + use_cols
    if DATE_FROM_COL in df.columns and DATE_THRU_COL in df.columns:
        use_cols = [DATE_FROM_COL, DATE_THRU_COL] + use_cols
    df = df[[c for c in use_cols if c in df.columns]].copy()

    df = df.dropna(subset=feats + [TARGET])
    df = df[(df[feats].isin([0, 1])).all(axis=1) & df[TARGET].isin([0, 1])]

    # Date-derived: Claim_Duration_Days, Claim_Month, Claim_Year (from CLM_FROM_DT)
    if DATE_FROM_COL in df.columns and DATE_THRU_COL in df.columns:
        from_dt = _parse_date(df[DATE_FROM_COL])
        thru_dt = _parse_date(df[DATE_THRU_COL])
        df["Claim_Duration_Days"] = (thru_dt - from_dt).dt.days
        df["Claim_Month"] = from_dt.dt.month
        df["Claim_Year"] = from_dt.dt.year
        # Drop rows where date parsing failed (NaT)
        df = df.dropna(subset=["Claim_Duration_Days", "Claim_Month", "Claim_Year"])
        date_feats = ["Claim_Duration_Days", "Claim_Month", "Claim_Year"]
    else:
        date_feats = []

    # SEGMENT one-hot
    segment_feats = []
    if SEGMENT_COL in df.columns:
        seg = df[SEGMENT_COL].fillna("_missing").astype(str)
        dummies = pd.get_dummies(seg, prefix="SEGMENT_", dtype=np.float32)
        segment_feats = list(dummies.columns)
        df = pd.concat([df.drop(columns=[SEGMENT_COL]), dummies], axis=1)

    # Assemble X: binary signals + segment dummies + date-derived
    all_feat_names = feats + segment_feats + date_feats
    X = df[all_feat_names].astype(np.float32)
    y = df[TARGET].astype(np.int64)
    return X, y, all_feat_names


def train_test_split_data(X, y):
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=(TEST_SIZE + VAL_SIZE), random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=(1 - val_ratio), random_state=RANDOM_STATE, stratify=y_rest
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def eval_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else 0.0
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }


if HAS_TORCH:
    class SmallMLP(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


if HAS_TORCH:
    def train_nn(X_train, y_train, X_val, y_val, n_features):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SmallMLP(n_features).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        Xt = torch.from_numpy(X_train.values if hasattr(X_train, "values") else X_train).float().to(device)
        yt = torch.from_numpy(y_train.values if hasattr(y_train, "values") else y_train).float().unsqueeze(1).to(device)
        Xv = torch.from_numpy(X_val.values if hasattr(X_val, "values") else X_val).float().to(device)
        yv = torch.from_numpy(y_val.values if hasattr(y_val, "values") else y_val).float().unsqueeze(1).to(device)

        best_val_loss = float("inf")
        best_state = None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(Xt), device=device)
            for i in range(0, len(Xt), BATCH_SIZE):
                idx = perm[i : i + BATCH_SIZE]
                opt.zero_grad()
                logits = model(Xt[idx])
                loss = criterion(logits, yt[idx].squeeze(1))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_logits = model(Xv)
                val_loss = criterion(val_logits, yv.squeeze(1)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        return model.to("cpu")

    def predict_nn(model, X):
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.values if hasattr(X, "values") else X).float()
            logits = model(x)
            probs = torch.sigmoid(logits).numpy()
        return probs, (probs >= 0.5).astype(np.int64)


def main():
    X, y, feats = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_data(X, y)
    n_features = len(feats)
    print(f"Features: {feats}")
    print(f"Train {len(X_train):,} / Val {len(X_val):,} / Test {len(X_test):,}")

    # Baseline: logistic regression
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    metrics_lr = eval_metrics(y_test, y_pred_lr, y_prob_lr)
    print("\n--- Baseline (Logistic Regression) ---")
    print(f"Accuracy: {metrics_lr['accuracy']:.4f}  Precision: {metrics_lr['precision']:.4f}  Recall: {metrics_lr['recall']:.4f}  F1: {metrics_lr['f1']:.4f}  ROC-AUC: {metrics_lr['roc_auc']:.4f}  PR-AUC: {metrics_lr['pr_auc']:.4f}")
    print("Confusion matrix:\n", metrics_lr["confusion_matrix"])

    # Neural network (if available)
    if HAS_TORCH:
        nn_model = train_nn(X_train, y_train, X_val, y_val, n_features)
        y_prob_nn, y_pred_nn = predict_nn(nn_model, X_test)
        metrics_nn = eval_metrics(y_test, y_pred_nn, y_prob_nn)
        print("\n--- Neural Network ---")
        print(f"Accuracy: {metrics_nn['accuracy']:.4f}  Precision: {metrics_nn['precision']:.4f}  Recall: {metrics_nn['recall']:.4f}  F1: {metrics_nn['f1']:.4f}  ROC-AUC: {metrics_nn['roc_auc']:.4f}  PR-AUC: {metrics_nn['pr_auc']:.4f}")
        print("Confusion matrix:\n", metrics_nn["confusion_matrix"])
    else:
        y_prob_nn = np.full(len(y_test), np.nan)
        y_pred_nn = np.full(len(y_test), -1)
        metrics_nn = None
        print("\n--- Neural Network skipped (torch not installed) ---")

    # Per-claim comparison table (test set)
    out = pd.DataFrame({
        "actual": y_test.values,
        "baseline_pred": y_pred_lr,
        "baseline_prob": y_prob_lr,
    })
    if HAS_TORCH:
        out["nn_pred"] = y_pred_nn
        out["nn_prob"] = y_prob_nn
    out.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nPredictions saved to {PREDICTIONS_CSV} (test set only)")

    # HTML comparison report
    rows = [
        ("Accuracy", metrics_lr["accuracy"], metrics_nn["accuracy"] if metrics_nn else None),
        ("Precision", metrics_lr["precision"], metrics_nn["precision"] if metrics_nn else None),
        ("Recall", metrics_lr["recall"], metrics_nn["recall"] if metrics_nn else None),
        ("F1", metrics_lr["f1"], metrics_nn["f1"] if metrics_nn else None),
        ("ROC-AUC", metrics_lr["roc_auc"], metrics_nn["roc_auc"] if metrics_nn else None),
        ("PR-AUC", metrics_lr["pr_auc"], metrics_nn["pr_auc"] if metrics_nn else None),
    ]

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Baseline vs NN</title>",
        "<style>table{border-collapse:collapse} th,td{border:1px solid #ccc; padding:6px 10px; text-align:left} th{background:#eee} .num{text-align:right}</style>",
        "</head><body><h1>Baseline vs Neural Network — Risk Model Comparison</h1>",
        f"<p>Test set: {len(y_test):,} claims. Features: {', '.join(feats)}.</p>",
        "<table><thead><tr><th>Metric</th><th class='num'>Baseline (Logistic Regression)</th><th class='num'>Neural Network</th></tr></thead><tbody>",
    ]
    for name, v_lr, v_nn in rows:
        nn_str = f"{v_nn:.4f}" if v_nn is not None else "—"
        html.append(f"<tr><td>{name}</td><td class='num'>{v_lr:.4f}</td><td class='num'>{nn_str}</td></tr>")
    html.append("</tbody></table>")
    html.append(f"<p><strong>Per-claim predictions:</strong> <code>{PREDICTIONS_CSV.name}</code> — columns: actual, baseline_pred, baseline_prob" + (", nn_pred, nn_prob" if HAS_TORCH else "") + ".</p>")
    html.append("</body></html>")
    COMPARISON_HTML.write_text("\n".join(html), encoding="utf-8")
    print(f"Comparison report: {COMPARISON_HTML}")


if __name__ == "__main__":
    main()
