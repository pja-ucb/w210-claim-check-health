from typing import Any, Dict, List, Tuple

from app.clients.local_model import LocalModelClient
from app.clients.rag import RagClient
from app.data_access import DataAccess
from app.models import ClaimInput, RiskEvidence, RiskResult


class RiskScorer:
    def __init__(
        self,
        data_access: DataAccess,
        model_client: LocalModelClient,
        rag_client: RagClient,
    ) -> None:
        self.data_access = data_access
        self.model_client = model_client
        self.rag_client = rag_client

    def _rule_score(self, claim: ClaimInput) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        score = 0.0
        fields = claim.fields

        claim_amount = _safe_float(fields.get("claim_amount"))

        if claim_amount is not None and claim_amount > 50000:
            score += 0.4
            reasons.append("High claim amount")

        if fields.get("flag_manual") is True:
            score += 0.1
            reasons.append("Manual review flag")

        score = min(score, 1.0)
        return score, reasons

    def score_claim(self, claim: ClaimInput, return_evidence: bool) -> RiskResult:
        enriched = self.data_access.enrich_claim({"claim_id": claim.claim_id, **claim.fields})
        claim.fields.update(enriched)

        rule_score, reasons = self._rule_score(claim)
        claim_type = claim.claim_type or claim.fields.get("claim_type") or "outpatient"
        model_score, threshold, debug = self.model_client.score(
            {"claim_id": claim.claim_id, **claim.fields},
            claim_type=str(claim_type).lower(),
        )

        risk_score = round((0.6 * model_score) + (0.4 * rule_score), 4)
        flag = model_score >= threshold or risk_score >= 0.7 or rule_score >= 0.8
        if model_score >= threshold:
            reasons.append("Model score above threshold")

        evidence: List[RiskEvidence] = []
        if return_evidence:
            query = ", ".join(reasons) if reasons else "General risk assessment"
            evidence = self.rag_client.retrieve(query)

        return RiskResult(
            claim_id=claim.claim_id,
            risk_score=risk_score,
            flag=flag,
            rule_score=round(rule_score, 4),
            model_score=round(model_score, 4),
            reasons=reasons or ["No rule triggers"],
            evidence=evidence,
            debug=debug,
        )


from typing import Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
