import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends

from python_service.api.dependencies import get_db_service, get_vectorizer_service
from python_service.api.schemas import HealthResponse, SearchRequest, SearchResponse
from python_service.db.service import RagDbService
from python_service.services.transform_service import MarkdownVectorizationService

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/search", response_model=SearchResponse, tags=["rag"])
async def search_documents(
    payload: SearchRequest,
    db: Annotated[RagDbService, Depends(get_db_service)],
    vectorizer: Annotated[MarkdownVectorizationService, Depends(get_vectorizer_service)],
) -> SearchResponse:
    query_vec = await asyncio.to_thread(vectorizer.vectorize_query, payload.query)
    results = await db.query_with_auto_merge(
        vector=query_vec,
        limit=payload.limit,
        threshold=payload.threshold,
    )
    return SearchResponse(results=results)
