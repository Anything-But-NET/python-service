from datetime import datetime
from enum import Enum
from typing import Any
import uuid

from pydantic import BaseModel, Field, field_validator, ConfigDict


class HealthStatus(str, Enum):
    UP = "UP"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


class ComponentHealth(BaseModel):
    status: HealthStatus
    details: dict[str, Any] | None = None


class HealthComponents(BaseModel):
    db: ComponentHealth
    vectorizer: ComponentHealth
    kafka: ComponentHealth
    llm: ComponentHealth


class HealthResponse(BaseModel):
    status: HealthStatus
    components: HealthComponents


class UserPermissions(BaseModel):
    allowed_groups: list[str] = Field(..., min_length=1)

    @field_validator("allowed_groups")
    @classmethod
    def validate_groups(cls, groups: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(
            group.strip()
            for group in groups
            if group and group.strip()
        ))
        if not normalized:
            raise ValueError("At least one allowed group is required")
        return normalized


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Текст пошукового запиту")
    user_permissions: UserPermissions
    limit: int = Field(default=20, ge=1, le=100)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class RagCitation(BaseModel):
    """Chunk reference as defined by ``RagCitation`` in contracts/openapi.yaml.

    Carries no display metadata: Core resolves title/url by document_id.
    """

    document_id: str
    chunk_id: str
    score: float


class SearchResultItem(BaseModel):
    point_id: str
    parent_id: str | None = None
    score: float
    children_count: int
    chunk_ids: list[str]
    chunks: list[str]
    document_id: str
    citations: list[RagCitation]
    text: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


class GenerationErrorStatus(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RETRIEVAL_TIMEOUT = "RETRIEVAL_TIMEOUT"
    VECTOR_DB_UNAVAILABLE = "VECTOR_DB_UNAVAILABLE"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class GenerationError(BaseModel):
    query_id: str | None = None
    status: GenerationErrorStatus
    message: str
    description: str | None = None
    timestamp: int


class RagGenerationStatus(str, Enum):
    ANSWER_GENERATED = "ANSWER_GENERATED"
    EMPTY_RETRIEVAL = "EMPTY_RETRIEVAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    ERROR = "ERROR"


class RagGenerationRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1, max_length=4000)
    user_permissions: UserPermissions

    @field_validator("query", "query_id")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be blank")
        return value


class RagGenerationResponse(BaseModel):
    query_id: str
    status: RagGenerationStatus
    answer: str | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    prompt_version_id: uuid.UUID | None = None
    citations: list[RagCitation]


class DashboardErrorCode(str, Enum):
    BAD_REQUEST = "BAD_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class DashboardErrorResponse(BaseModel):
    error_code: DashboardErrorCode
    message: str
    timestamp: datetime


class TokenIntrospectResponse(BaseModel):
    active: bool
    operator_id: uuid.UUID | None = None
    email: str | None = None
    status: str | None = None
    authorities: list[str] = Field(default_factory=list)


class PromptVersionSummary(BaseModel):
    id: uuid.UUID
    created_at: datetime


class PromptVersionDetail(BaseModel):
    id: uuid.UUID
    prompt_text: str
    metadata: dict[str, Any]
    created_at: datetime


class PromptVersionsPage(BaseModel):
    items: list[PromptVersionSummary]
    page: int = Field(ge=0)
    size: int = Field(ge=1)
    total_elements: int = Field(ge=0)
    total_pages: int = Field(ge=0)

class PromptMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)


class EvaluationRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ThresholdDecisionStatus(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"


class ExpectedOutcomeType(str, Enum):
    ANSWER_EXPECTED = "ANSWER_EXPECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class QualityDimension(str, Enum):
    RELEVANCE = "relevance"
    FACTUAL_GROUNDING = "factual_grounding"
    CITATION_CORRECTNESS = "citation_correctness"
    SAFE_NO_ANSWER = "safe_no_answer"


class QualityThresholdsConfig(BaseModel):
    decision_status: ThresholdDecisionStatus
    relevance_min_score: float = Field(..., ge=0.0, le=1.0)
    factual_grounding_min_score: float = Field(..., ge=0.0, le=1.0)
    citation_correctness_min_score: float = Field(..., ge=0.0, le=1.0)
    target_pass_rate: float = Field(..., ge=0.0, le=1.0)


class EvaluationDatasetConfig(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    dataset_path: str | None = None
    known_limitations: list[str] = Field(default_factory=list)


class EvaluationRunCreateRequest(BaseModel):
    prompt_version_id: uuid.UUID
    dataset: EvaluationDatasetConfig
    thresholds: QualityThresholdsConfig


class QualityCheckResult(BaseModel):
    applicable: bool
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    calculation_method: str = Field(..., min_length=1)
    details: dict[str, Any] | None = None


class CaseQualityChecks(BaseModel):
    relevance: QualityCheckResult
    factual_grounding: QualityCheckResult
    citation_correctness: QualityCheckResult
    safe_no_answer: QualityCheckResult


class CaseEvaluationResult(BaseModel):
    case_id: str
    query: str
    expected_outcome: ExpectedOutcomeType
    expected_document_ids: list[str] = Field(default_factory=list)
    actual_status: RagGenerationStatus
    actual_answer: str | None = None
    actual_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    actual_citations: list[RagCitation] = Field(default_factory=list)
    passed: bool
    checks: CaseQualityChecks
    failure_reasons: list[str] = Field(default_factory=list)


class DimensionAggregateSummary(BaseModel):
    dimension: QualityDimension
    evaluated_cases_count: int = Field(ge=0)
    passed_cases_count: int = Field(ge=0)
    observed_pass_rate: float = Field(ge=0.0, le=1.0)
    observed_mean_score: float = Field(ge=0.0, le=1.0)
    proposed_threshold_score: float = Field(ge=0.0, le=1.0)
    proposed_target_pass_rate: float = Field(ge=0.0, le=1.0)
    threshold_decision_status: ThresholdDecisionStatus
    meets_proposed_threshold: bool


class EvaluationRunMetadataPayload(BaseModel):
    dataset: EvaluationDatasetConfig
    thresholds: QualityThresholdsConfig
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    overall_pass_rate: float = Field(ge=0.0, le=1.0)
    dimension_summaries: list[DimensionAggregateSummary]
    followup_jira_issues: list[str] = Field(default_factory=list)
    case_results: list[CaseEvaluationResult]


class EvaluationRunResponse(BaseModel):
    id: uuid.UUID
    prompt_version_id: uuid.UUID
    status: EvaluationRunStatus
    metadata: EvaluationRunMetadataPayload
    created_at: datetime

