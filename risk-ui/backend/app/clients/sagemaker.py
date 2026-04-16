import hashlib
from typing import Dict, Any


class SageMakerClient:
    """Stubbed SageMaker client.

    Replace `score` with a real invocation to a SageMaker endpoint.
    """

    def __init__(self, endpoint_name: str | None = None) -> None:
        self.endpoint_name = endpoint_name or "mock-endpoint"

    def score(self, payload: Dict[str, Any]) -> float:
        # Deterministic placeholder score based on claim_id hash.
        claim_id = str(payload.get("claim_id", ""))
        h = hashlib.sha256(claim_id.encode("utf-8")).hexdigest()
        # Convert hash prefix to [0, 1] range
        return (int(h[:8], 16) % 1000) / 1000.0
