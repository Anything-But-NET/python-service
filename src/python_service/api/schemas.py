from enum import Enum

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"


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
    confidence_score: float = Field(ge=0.0, le=1.0)
    citations: list[RagCitation]
