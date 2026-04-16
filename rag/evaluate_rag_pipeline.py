from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from langchain_openai import ChatOpenAI


load_dotenv()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()



def normalize_spaces(text: str) -> str:
    return " ".join(clean_text(text).split())



def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        val = normalize_spaces(item)
        if not val:
            continue
        key = val.lower()
        if key not in seen:
            seen.add(key)
            out.append(val)
    return out



def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


# ------------------------------------------------------------
# Flatten pipeline output into RAGAS friendly fields
# ------------------------------------------------------------


def flatten_summary_text(summary: Dict[str, Any]) -> str:
    gaps = summary.get("main_gaps") or []
    if not isinstance(gaps, list):
        gaps = [clean_text(gaps)] if clean_text(gaps) else []

    parts = [
        f"Service summary: {clean_text(summary.get('service_summary'))}",
        f"Medical necessity: {clean_text(summary.get('medical_necessity_findings'))}",
        f"Billing/coding: {clean_text(summary.get('billing_coding_findings'))}",
        f"Documentation: {clean_text(summary.get('documentation_findings'))}",
        f"Limitations: {clean_text(summary.get('limitations_findings'))}",
        f"Evidence strength: {clean_text(summary.get('evidence_strength'))}",
        f"Policy ambiguity: {clean_text(summary.get('policy_ambiguity'))}",
    ]

    if gaps:
        parts.append(f"Main gaps: {', '.join(dedupe_keep_order(gaps))}")

    return "\n".join([p for p in parts if clean_text(p)])



def flatten_recommended_action_text(recommended_action: Dict[str, Any]) -> str:
    secondary = recommended_action.get("secondary_actions") or []
    if not isinstance(secondary, list):
        secondary = [clean_text(secondary)] if clean_text(secondary) else []

    parts = [
        f"Primary action: {clean_text(recommended_action.get('primary_action'))}",
        f"Secondary actions: {', '.join(dedupe_keep_order(secondary)) if secondary else 'None'}",
        f"Decision rationale: {clean_text(recommended_action.get('decision_rationale'))}",
        f"Claim summary: {clean_text(recommended_action.get('claim_summary'))}",
        f"Driver policy alignment: {clean_text(recommended_action.get('driver_policy_alignment'))}",
    ]
    return "\n".join([p for p in parts if clean_text(p)])



def extract_selected_contexts(hcpc_result: Dict[str, Any]) -> List[str]:
    contexts: List[str] = []
    for chunk in hcpc_result.get("selected_chunks", []) or []:
        text = (
            clean_text(chunk.get("text"))
            or clean_text(chunk.get("text_preview"))
        )
        if text:
            contexts.append(text)
    return dedupe_keep_order(contexts)



def build_hcpc_question(
    claim_id: str,
    hcpc: str,
    diag_descs: Optional[Sequence[str]] = None,
    proc_descs: Optional[Sequence[str]] = None,
) -> str:
    diag_text = ", ".join(dedupe_keep_order(diag_descs or [])[:3]) or "the claim diagnoses"
    proc_text = ", ".join(dedupe_keep_order(proc_descs or [])[:3]) or "the claim procedures"
    return (
        f"For claim {claim_id} and HCPCS {hcpc}, what does the retrieved CMS policy evidence say "
        f"about medical necessity, billing and coding, documentation, and limitations, given diagnoses "
        f"such as {diag_text} and procedures such as {proc_text}?"
    )



def build_action_question(claim_id: str) -> str:
    return (
        f"For claim {claim_id}, what is the most appropriate next review action based on the retrieved "
        f"policy evidence and the claim-level risk signals?"
    )


# ------------------------------------------------------------
# Gold/reference loading
# ------------------------------------------------------------


def load_reference_map(reference_json: Optional[Path]) -> Dict[Tuple[str, str], str]:
    if reference_json is None or not reference_json.exists():
        return {}

    obj = load_json(reference_json)
    records = obj if isinstance(obj, list) else obj.get("records", [])

    ref_map: Dict[Tuple[str, str], str] = {}
    for rec in records:
        claim_id = clean_text(rec.get("claim_id"))
        hcpc = clean_text(rec.get("hcpc"))
        reference = clean_text(rec.get("reference"))
        if claim_id and hcpc and reference:
            ref_map[(claim_id, hcpc)] = reference
    return ref_map



def load_action_label_map(action_labels_json: Optional[Path]) -> Dict[str, str]:
    if action_labels_json is None or not action_labels_json.exists():
        return {}

    obj = load_json(action_labels_json)
    records = obj if isinstance(obj, list) else obj.get("records", [])

    label_map: Dict[str, str] = {}
    for rec in records:
        claim_id = clean_text(rec.get("claim_id"))
        label = clean_text(rec.get("primary_action"))
        if claim_id and label:
            label_map[claim_id] = label
    return label_map


# ------------------------------------------------------------
# Dataset builders
# ------------------------------------------------------------


def build_hcpc_eval_rows(
    output_json: Dict[str, Any],
    reference_map: Optional[Dict[Tuple[str, str], str]] = None,
) -> List[Dict[str, Any]]:
    claim_id = clean_text(output_json.get("claim_id"))
    per_hcpc_results = output_json.get("per_hcpc_results", []) or []

    diagnosis_descriptions = output_json.get("diagnosis_descriptions", []) or []
    procedure_descriptions = output_json.get("procedure_descriptions", []) or []

    rows: List[Dict[str, Any]] = []
    for item in per_hcpc_results:
        hcpc = clean_text(item.get("hcpc"))
        if not hcpc:
            continue

        summary = item.get("summary", {}) or {}
        contexts = extract_selected_contexts(item)
        response = flatten_summary_text(summary)
        reference = ""
        if reference_map:
            reference = reference_map.get((claim_id, hcpc), "")

        rows.append(
            {
                "claim_id": claim_id,
                "hcpc": hcpc,
                "user_input": build_hcpc_question(
                    claim_id=claim_id,
                    hcpc=hcpc,
                    diag_descs=diagnosis_descriptions,
                    proc_descs=procedure_descriptions,
                ),
                "retrieved_contexts": contexts,
                "response": response,
                "reference": reference,
                "evidence_strength": clean_text(summary.get("evidence_strength")),
                "policy_ambiguity": clean_text(summary.get("policy_ambiguity")),
                "retrieved_doc_count": item.get("retrieved_doc_count", 0),
                "reranked_doc_count": item.get("reranked_doc_count", 0),
                "selected_chunk_count": item.get("selected_chunk_count", 0),
                "article_ids": item.get("article_ids", []),
            }
        )

    return rows



def build_action_eval_rows(
    output_json: Dict[str, Any],
    action_label_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    claim_id = clean_text(output_json.get("claim_id"))
    recommended_action = output_json.get("recommended_action", {}) or {}
    per_hcpc_results = output_json.get("per_hcpc_results", []) or []

    combined_contexts: List[str] = []
    for item in per_hcpc_results:
        combined_contexts.extend(extract_selected_contexts(item))
    combined_contexts = dedupe_keep_order(combined_contexts)

    response = flatten_recommended_action_text(recommended_action)
    reference = ""
    if action_label_map:
        reference = action_label_map.get(claim_id, "")

    return [
        {
            "claim_id": claim_id,
            "user_input": build_action_question(claim_id),
            "retrieved_contexts": combined_contexts,
            "response": response,
            "reference": reference,
            "predicted_primary_action": clean_text(recommended_action.get("primary_action")),
            "predicted_secondary_actions": recommended_action.get("secondary_actions", []),
            "risk_probability": to_float(output_json.get("risk_probability")),
            "top_model_risk_drivers": output_json.get("top_model_risk_drivers", []),
        }
    ]



def load_outputs_from_dir(outputs_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(outputs_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {outputs_dir}")
    return [load_json(fp) for fp in files]



def build_ragas_dataset(rows: List[Dict[str, Any]], include_reference: bool = True) -> Dataset:
    dataset_rows: List[Dict[str, Any]] = []
    for row in rows:
        record = {
            "user_input": row["user_input"],
            "retrieved_contexts": row["retrieved_contexts"],
            "response": row["response"],
        }
        if include_reference and clean_text(row.get("reference")):
            record["reference"] = row["reference"]
        dataset_rows.append(record)
    return Dataset.from_list(dataset_rows)


# ------------------------------------------------------------
#  RAGAS execution
# ------------------------------------------------------------


def run_ragas(
    dataset: Dataset,
    evaluator_model: str = "gpt-4o-mini",
    use_reference_metrics: bool = False,
) -> pd.DataFrame:

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=evaluator_model, temperature=0))

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
    ]

    has_reference = use_reference_metrics and "reference" in dataset.column_names

    if has_reference:
        # Reference-based retrieval metrics
        try:
            from ragas.metrics import ContextPrecision
            metrics.append(ContextPrecision())
        except Exception:
            pass

        try:
            from ragas.metrics import LLMContextRecall
            metrics.append(LLMContextRecall())
        except Exception:
            pass

        try:
            from ragas.metrics import SemanticSimilarity
            metrics.append(SemanticSimilarity())
        except Exception:
            pass
    else:
        # No-reference setup: avoid reference-dependent metrics
        try:
            from ragas.metrics import ContextUtilization
            metrics.append(ContextUtilization())
        except Exception:
            pass

    results = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)
    return results.to_pandas()



# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally run RAGAS evaluation for claim policy pipeline outputs.")
    parser.add_argument(
        "--outputs-dir",
        type=str,
        required=True,
        help="Directory containing claim_policy_output JSON files.",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="hcpc",
        choices=["hcpc", "action", "both"],
        help="What evaluation rows to build.",
    )
    parser.add_argument(
        "--reference-json",
        type=str,
        default=None,
        help="Optional JSON file with HCPC-level reference summaries. Expected keys: claim_id, hcpc, reference.",
    )
    parser.add_argument(
        "--action-labels-json",
        type=str,
        default=None,
        help="Optional JSON file with claim-level gold primary actions. Expected keys: claim_id, primary_action.",
    )
    parser.add_argument(
        "--write-dir",
        type=str,
        default="./ragas_eval_outputs",
        help="Directory to write datasets/results.",
    )
    parser.add_argument(
        "--run-ragas",
        action="store_true",
        help="Actually run RAGAS after building the dataset.",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model used by RAGAS if --run-ragas is set.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser()
    write_dir = Path(args.write_dir).expanduser()
    write_dir.mkdir(parents=True, exist_ok=True)

    reference_map = load_reference_map(Path(args.reference_json).expanduser()) if args.reference_json else {}
    action_label_map = load_action_label_map(Path(args.action_labels_json).expanduser()) if args.action_labels_json else {}

    output_jsons = load_outputs_from_dir(outputs_dir)

    hcpc_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []

    for obj in output_jsons:
        if args.level in {"hcpc", "both"}:
            hcpc_rows.extend(build_hcpc_eval_rows(obj, reference_map=reference_map))
        if args.level in {"action", "both"}:
            action_rows.extend(build_action_eval_rows(obj, action_label_map=action_label_map))

    if hcpc_rows:
        hcpc_df = pd.DataFrame(hcpc_rows)
        hcpc_df.to_json(write_dir / "hcpc_eval_rows.json", orient="records", indent=2)
        hcpc_df.to_csv(write_dir / "hcpc_eval_rows.csv", index=False)

        hcpc_dataset = build_ragas_dataset(hcpc_rows, include_reference=True)
        hcpc_dataset.to_json(str(write_dir / "hcpc_ragas_dataset.json"))
        print(f"Built HCPC dataset with {len(hcpc_dataset)} rows")

        if args.run_ragas:
            hcpc_results_df = run_ragas(
                dataset=hcpc_dataset,
                evaluator_model=args.evaluator_model,
                use_reference_metrics=bool(reference_map),
            )
            hcpc_results_df.to_csv(write_dir / "hcpc_ragas_results.csv", index=False)
            print("Saved HCPC RAGAS results")

    if action_rows:
        action_df = pd.DataFrame(action_rows)
        action_df.to_json(write_dir / "action_eval_rows.json", orient="records", indent=2)
        action_df.to_csv(write_dir / "action_eval_rows.csv", index=False)

        action_dataset = build_ragas_dataset(action_rows, include_reference=True)
        action_dataset.to_json(str(write_dir / "action_ragas_dataset.json"))
        print(f"Built action dataset with {len(action_dataset)} rows")

        if args.run_ragas:
            action_results_df = run_ragas(
                dataset=action_dataset,
                evaluator_model=args.evaluator_model,
                use_reference_metrics=False,
            )
            action_results_df.to_csv(write_dir / "action_ragas_results.csv", index=False)
            print("Saved action RAGAS results")


    if not hcpc_rows and not action_rows:
        raise ValueError("No evaluation rows were built. Check your outputs directory and JSON structure.")


if __name__ == "__main__":
    main()
