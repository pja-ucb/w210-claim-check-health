from typing import Dict, Any


class DataAccess:
    """Stub data access layer.

    Replace with Athena/RDS/S3 integrations to fetch claim/policy details.
    """

    def __init__(self) -> None:
        pass

    def enrich_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        # No-op placeholder
        return claim
