from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClaimInput(BaseModel):
    claim_id: str
    policy_id: Optional[str] = None
    claim_type: Optional[str] = None
    fields: Dict[str, Any] = Field(default_factory=dict)


class RiskEvidence(BaseModel):
    source: str
    snippet: str


class RiskResult(BaseModel):
    claim_id: str
    risk_score: float
    flag: bool
    rule_score: float
    model_score: float
    reasons: List[str]
    evidence: List[RiskEvidence] = Field(default_factory=list)
    debug: Optional[Dict[str, Any]] = None


class BatchRequest(BaseModel):
    claims: List[ClaimInput]
    return_evidence: bool = True


class BatchSummary(BaseModel):
    total: int
    flagged: int
    flagged_rate: float


class BatchResponse(BaseModel):
    results: List[RiskResult]
    summary: BatchSummary


class PolicyReviewRequest(BaseModel):
    claim_id: str


class PolicyRecommendedAction(BaseModel):
    """Layer-2 output: allowed actions are enforced in the pipeline."""

    model_config = ConfigDict(extra="allow")

    primary_action: str = ""
    secondary_actions: List[str] = Field(default_factory=list)
    decision_rationale: str = ""
    claim_summary: str = ""
    claim_action_confidence: str = ""
    action_drivers: List[str] = Field(default_factory=list)


class PolicyHcpcSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    hcpc: Optional[str] = None
    service_summary: Optional[str] = None
    medical_necessity_findings: Optional[str] = None
    billing_coding_findings: Optional[str] = None
    documentation_findings: Optional[str] = None
    limitations_findings: Optional[str] = None
    evidence_strength: Optional[str] = None
    policy_ambiguity: Optional[str] = None
    main_gaps: List[str] = Field(default_factory=list)


class PolicyPerHcpcResult(BaseModel):
    """One HCPCS path through retrieval + Layer-1 summary."""

    model_config = ConfigDict(extra="allow")

    hcpc: str = ""
    article_ids: List[Any] = Field(default_factory=list)
    summary: Optional[PolicyHcpcSummary] = None


class PolicyReviewResponse(BaseModel):
    """
    Full pipeline JSON returned to the UI.
    Extra keys from the pipeline are preserved (e.g. hcpcs_selection_scores).
    """

    model_config = ConfigDict(extra="allow")

    claim_id: str = ""
    claim_year: Optional[int] = None
    risk_probability: Optional[float] = None
    risk_label: Optional[str] = None
    upstream_high_risk_flag: Optional[bool] = None
    top_model_risk_drivers: List[str] = Field(default_factory=list)
    all_hcpcs: List[str] = Field(default_factory=list)
    selected_primary_hcpcs: List[str] = Field(default_factory=list)
    hcpcs: List[str] = Field(default_factory=list)
    diagnosis_codes: List[str] = Field(default_factory=list)
    diagnosis_descriptions: List[str] = Field(default_factory=list)
    procedure_codes: List[str] = Field(default_factory=list)
    procedure_descriptions: List[str] = Field(default_factory=list)
    per_hcpc_results: List[PolicyPerHcpcResult] = Field(default_factory=list)
    recommended_action: PolicyRecommendedAction
    per_hcpc_recommended_followup: List[Dict[str, Any]] = Field(default_factory=list)
