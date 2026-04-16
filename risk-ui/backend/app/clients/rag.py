from typing import List, Optional

from app.models import RiskEvidence


class RagClient:
    """Stubbed RAG client.

    Replace `retrieve` with a real vector search or RAG endpoint call.
    """

    def __init__(self, endpoint_url: Optional[str] = None) -> None:
        self.endpoint_url = endpoint_url or "mock-rag-endpoint"

    def retrieve(self, query: str) -> List[RiskEvidence]:
        return [
            RiskEvidence(source="policy_guide", snippet=f"Relevant policy: {query}"),
            RiskEvidence(source="claims_history", snippet="Similar claims show elevated risk."),
        ]
