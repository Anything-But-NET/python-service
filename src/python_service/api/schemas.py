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


class SearchResultItem(BaseModel):
    point_id: str
    parent_id: str | None = None
    score: float
    children_count: int
    chunk_ids: list[str]
    chunks: list[str]
    document_id: str | None = None
    text: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]