from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Текст пошукового запиту")
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
