"""
Runs the Medicare policy RAG pipeline from the repo `RAG/` folder.

Paths default to `<w210>/RAG/...` and can be overridden with env vars:
  RAG_DIR, RAG_CLAIMS_CSV, RAG_CMS_HCPC_CSV, RAG_CHROMA_PATH,
  RAG_DIAGNOSIS_LOOKUP_CSV, RAG_PROCEDURE_LOOKUP_CSV,
  RAG_CHROMA_COLLECTION, RAG_LLM_MODEL, RAG_EMBEDDING_MODEL
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# w210/risk_ui/backend/app/this_file.py -> four parents up = w210 root
_W210_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_RAG_DIR = _W210_ROOT / "RAG"


def _rag_paths() -> Dict[str, str]:
    rag = Path(os.environ.get("RAG_DIR", str(_DEFAULT_RAG_DIR))).expanduser().resolve()
    data = rag / "data"
    default_claims = (
        _W210_ROOT
        / "nn"
        / "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1_labeled_rag_default.csv"
    )
    return {
        "claims_csv": os.environ.get("RAG_CLAIMS_CSV", str(default_claims)),
        "cms_hcpc_csv": os.environ.get("RAG_CMS_HCPC_CSV", str(data / "cmd_hcpc_data.csv")),
        "chroma_path": os.environ.get("RAG_CHROMA_PATH", str(rag / "medicare_chroma_database")),
        "diagnosis_lookup_csv": os.environ.get(
            "RAG_DIAGNOSIS_LOOKUP_CSV", str(data / "icd9_diagnosis_lookup.csv")
        ),
        "procedure_lookup_csv": os.environ.get(
            "RAG_PROCEDURE_LOOKUP_CSV", str(data / "icd9_procedure_lookup.csv")
        ),
    }


def _load_pipeline_module():
    rag_dir = Path(os.environ.get("RAG_DIR", str(_DEFAULT_RAG_DIR))).expanduser().resolve()
    script = rag_dir / "claim_policy_pipeline_updated_full.py"
    if not script.is_file():
        raise FileNotFoundError(f"RAG pipeline script not found: {script}")

    spec = importlib.util.spec_from_file_location("claim_policy_pipeline", script)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load claim_policy_pipeline spec")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses / imports resolve
    sys.modules["claim_policy_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


_pipeline_mod = None


def get_pipeline_module():
    global _pipeline_mod
    if _pipeline_mod is None:
        _pipeline_mod = _load_pipeline_module()
    return _pipeline_mod


def run_policy_review(claim_id: str) -> Dict[str, Any]:
    """
    Execute full RAG pipeline for one claim. Requires OPENAI_API_KEY and local data files.
    """
    claim_id = str(claim_id).strip()
    if not claim_id:
        raise ValueError("claim_id is required")

    mod = get_pipeline_module()
    paths = _rag_paths()

    # Ensure pipeline's load_dotenv sees RAG/.env
    try:
        from dotenv import load_dotenv

        rag_dir = Path(os.environ.get("RAG_DIR", str(_DEFAULT_RAG_DIR))).expanduser().resolve()
        load_dotenv(rag_dir / ".env")
        load_dotenv(_W210_ROOT / ".env")
    except ImportError:
        pass

    collection = os.environ.get("RAG_CHROMA_COLLECTION", mod.DEFAULT_CHROMA_COLLECTION)
    llm_model = os.environ.get("RAG_LLM_MODEL", mod.DEFAULT_LLM_MODEL)
    embedding_model = os.environ.get("RAG_EMBEDDING_MODEL", mod.DEFAULT_EMBEDDING_MODEL)

    config = mod.PipelineConfig(
        claims_csv=paths["claims_csv"],
        cms_hcpc_csv=paths["cms_hcpc_csv"],
        chroma_path=paths["chroma_path"],
        chroma_collection=collection,
        diagnosis_lookup_csv=paths.get("diagnosis_lookup_csv"),
        procedure_lookup_csv=paths.get("procedure_lookup_csv"),
        llm_model=llm_model,
        embedding_model=embedding_model,
        max_claim_hcpcs=int(os.environ.get("RAG_MAX_CLAIM_HCPCS", "2")),
        hcpcs_prefilter_candidates=int(os.environ.get("RAG_HCPCS_PREFILTER", "4")),
    )

    return mod.run_pipeline(claim_id=claim_id, config=config)


def policy_review_health() -> Tuple[bool, str]:
    """Quick check that RAG script and key paths exist."""
    try:
        paths = _rag_paths()
        rag_dir = Path(os.environ.get("RAG_DIR", str(_DEFAULT_RAG_DIR))).expanduser().resolve()
        script = rag_dir / "claim_policy_pipeline_updated_full.py"
        if not script.is_file():
            return False, f"Missing pipeline script: {script}"
        for key in ("claims_csv", "cms_hcpc_csv", "chroma_path"):
            p = Path(paths[key])
            if not p.exists():
                return False, f"Missing {key}: {p}"
        return True, "ok"
    except Exception as e:
        return False, str(e)
