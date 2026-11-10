from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from python_service.api.routes import router
from python_service.api.schemas import RagGenerationResponse, RagGenerationStatus


@pytest.mark.asyncio
async def test_generate_endpoint_matches_java_rag_contract():
    service = MagicMock()
    service.generate = AsyncMock(return_value=RagGenerationResponse(
        query_id="query-123",
        status=RagGenerationStatus.ANSWER_GENERATED,
        answer="Submit the request through the HR portal.",
        confidence_score=0.91,
        citations=[{
            "document_id": "document-1",
            "chunk_id": "chunk-1",
            "score": 0.91,
        }],
    ))

    app = FastAPI()
    app.include_router(router)
    app.state.grounded_answer_service = service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/generate",
            json={
                "query_id": "query-123",
                "query": "How do I request annual leave?",
                "user_permissions": {"allowed_groups": ["employees"]},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "query_id": "query-123",
        "status": "ANSWER_GENERATED",
        "answer": "Submit the request through the HR portal.",
        "confidence_score": 0.91,
        "citations": [{
            "document_id": "document-1",
            "chunk_id": "chunk-1",
            "score": 0.91,
        }],
    }
    service.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_endpoint_rejects_empty_permission_groups():
    service = MagicMock()
    service.generate = AsyncMock()

    app = FastAPI()
    app.include_router(router)
    app.state.grounded_answer_service = service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/generate",
            json={
                "query_id": "query-123",
                "query": "How do I request annual leave?",
                "user_permissions": {"allowed_groups": ["", "  "]},
            },
        )

    assert response.status_code == 422
    service.generate.assert_not_awaited()
