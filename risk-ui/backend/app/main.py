from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os

from app.clients.local_model import LocalModelClient
from app.clients.rag import RagClient
from app.data_access import DataAccess
from app.models import (
    BatchRequest,
    BatchResponse,
    BatchSummary,
    PolicyReviewRequest,
    PolicyReviewResponse,
)
from app.policy_review_service import policy_review_health, run_policy_review
from app.risk import RiskScorer


app = FastAPI(title="Risk Scoring API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

outpatient_dir = os.environ.get("OUTPATIENT_MODEL_DIR", "/Users/pedro.josealvarez/w210/nn")
inpatient_dir = os.environ.get("INPATIENT_MODEL_DIR")

scorer = RiskScorer(
    data_access=DataAccess(),
    model_client=LocalModelClient(outpatient_dir=outpatient_dir, inpatient_dir=inpatient_dir),
    rag_client=RagClient(),
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/policy-review/health")
def policy_review_health_check() -> dict:
    ok, msg = policy_review_health()
    return {"rag_ready": ok, "message": msg}


@app.post("/policy-review", response_model=PolicyReviewResponse)
def policy_review(req: PolicyReviewRequest) -> PolicyReviewResponse:
    """
    Run the full Medicare policy RAG pipeline for a single claim_id.
    Can take several minutes (LLM + vector retrieval). Requires OPENAI_API_KEY.
    """
    try:
        raw = run_policy_review(req.claim_id)
        return PolicyReviewResponse.model_validate(raw)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/risk-score-batch", response_model=BatchResponse)
def risk_score_batch(req: BatchRequest) -> BatchResponse:
    results = [scorer.score_claim(c, req.return_evidence) for c in req.claims]
    total = len(results)
    flagged = sum(1 for r in results if r.flag)
    summary = BatchSummary(total=total, flagged=flagged, flagged_rate=(flagged / total) if total else 0.0)
    return BatchResponse(results=results, summary=summary)


@app.post("/debug-model")
def debug_model(req: BatchRequest) -> dict:
    debug_results = []
    for claim in req.claims:
        claim_type = claim.claim_type or claim.fields.get("claim_type") or "outpatient"
        model_score, threshold, debug = scorer.model_client.score(
            {"claim_id": claim.claim_id, **claim.fields},
            claim_type=str(claim_type).lower(),
        )
        debug_results.append(
            {
                "claim_id": claim.claim_id,
                "claim_type": claim_type,
                "model_score": model_score,
                "threshold": threshold,
                "debug": debug,
            }
        )
    return {"results": debug_results}
