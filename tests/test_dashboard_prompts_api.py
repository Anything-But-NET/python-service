from datetime import UTC, datetime
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from python_service.api.dependencies import get_auth_client, get_db_service
from python_service.api.errors import DashboardApiError
from python_service.api.schemas import (
    DashboardErrorCode,
    RagGenerationRequest,
    RagGenerationStatus,
    TokenIntrospectResponse,
    UserPermissions,
)
from python_service.config import Settings
from python_service.main import create_app
from python_service.models.domain import PromptVersion
from python_service.services.auth_client import CoreAuthClient
from python_service.services.grounded_answer_service import GroundedAnswerService


def _build_client(db_mock: MagicMock, auth_mock: MagicMock) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_db_service] = lambda: db_mock
    app.dependency_overrides[get_auth_client] = lambda: auth_mock
    return TestClient(app, raise_server_exceptions=False)


def test_list_prompts_returns_paginated_summaries() -> None:
    prompt_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
    created_at = datetime(2026, 11, 19, 14, 25, 3, tzinfo=UTC)

    db_mock = MagicMock()
    db_mock.list_prompt_versions = AsyncMock(
        return_value=(
            [PromptVersion(id=prompt_id, prompt_text="sys", metadata={"model": "gpt"}, created_at=created_at)],
            1,
        )
    )
    auth_mock = MagicMock()
    auth_mock.introspect = AsyncMock(
        return_value=TokenIntrospectResponse(
            active=True,
            operator_id=uuid.uuid4(),
            email="oleh@company.com",
            status="ACTIVE",
            authorities=["PROMPTS"],
        )
    )

    client = _build_client(db_mock, auth_mock)
    resp = client.get(
        "/api/v1/dash/prompts?page=0&size=20",
        headers={"Authorization": "Bearer valid.jwt"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["page"] == 0
    assert body["size"] == 20
    assert body["total_elements"] == 1
    assert body["total_pages"] == 1
    assert body["items"] == [
        {"id": str(prompt_id), "created_at": "2026-11-19T14:25:03Z"}
    ]
    auth_mock.introspect.assert_awaited_once_with("valid.jwt")
    db_mock.list_prompt_versions.assert_awaited_once_with(page=0, size=20)


def test_list_prompts_auth_and_validation_errors() -> None:
    db_mock = MagicMock()
    auth_mock = MagicMock()
    client = _build_client(db_mock, auth_mock)

    # Missing Authorization header -> 401
    resp = client.get("/api/v1/dash/prompts")
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "UNAUTHORIZED"

    # Inactive token -> 401
    auth_mock.introspect = AsyncMock(return_value=TokenIntrospectResponse(active=False, authorities=[]))
    resp = client.get("/api/v1/dash/prompts", headers={"Authorization": "Bearer expired.jwt"})
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "UNAUTHORIZED"

    # Active token without PROMPTS authority -> 403
    auth_mock.introspect = AsyncMock(
        return_value=TokenIntrospectResponse(
            active=True,
            operator_id=uuid.uuid4(),
            email="logs@company.com",
            status="ACTIVE",
            authorities=["LOGS", "ANALYTICS"],
        )
    )
    resp = client.get("/api/v1/dash/prompts", headers={"Authorization": "Bearer logs.jwt"})
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "FORBIDDEN"

    # Invalid pagination param -> 400 BAD_REQUEST
    auth_mock.introspect = AsyncMock(
        return_value=TokenIntrospectResponse(
            active=True,
            operator_id=uuid.uuid4(),
            email="prompts@company.com",
            status="ACTIVE",
            authorities=["PROMPTS"],
        )
    )
    resp = client.get("/api/v1/dash/prompts?size=500", headers={"Authorization": "Bearer ok.jwt"})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "BAD_REQUEST"


def test_get_prompt_version_by_id_returns_detail_404_and_400() -> None:
    prompt_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
    created_at = datetime(2026, 11, 19, 14, 25, 3, tzinfo=UTC)

    db_mock = MagicMock()
    db_mock.get_prompt_version = AsyncMock(
        return_value=PromptVersion(
            id=prompt_id,
            prompt_text="Answer strictly from context.",
            metadata={"model": "gpt-4o-mini", "version": "v1.2"},
            created_at=created_at,
        )
    )
    auth_mock = MagicMock()
    auth_mock.introspect = AsyncMock(
        return_value=TokenIntrospectResponse(
            active=True,
            operator_id=uuid.uuid4(),
            email="prompts@company.com",
            status="ACTIVE",
            authorities=["PROMPTS"],
        )
    )

    client = _build_client(db_mock, auth_mock)
    resp = client.get(
        f"/api/v1/dash/prompts/{prompt_id}",
        headers={"Authorization": "Bearer ok.jwt"},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "id": str(prompt_id),
        "prompt_text": "Answer strictly from context.",
        "metadata": {"model": "gpt-4o-mini", "version": "v1.2"},
        "created_at": "2026-11-19T14:25:03Z",
    }

    # Not found -> 404
    db_mock.get_prompt_version = AsyncMock(return_value=None)
    missing_id = uuid.uuid4()
    resp_404 = client.get(
        f"/api/v1/dash/prompts/{missing_id}",
        headers={"Authorization": "Bearer ok.jwt"},
    )
    assert resp_404.status_code == 404
    assert resp_404.json()["error_code"] == "NOT_FOUND"

    # Invalid UUID -> 400
    resp_400 = client.get(
        "/api/v1/dash/prompts/not-a-uuid",
        headers={"Authorization": "Bearer ok.jwt"},
    )
    assert resp_400.status_code == 400
    assert resp_400.json()["error_code"] == "BAD_REQUEST"


async def test_core_auth_client_introspect_success_and_error() -> None:
    operator_id = uuid.uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/internal/auth/introspect"
        payload = json.loads(request.content)
        if payload["token"] == "good":
            return httpx.Response(
                200,
                json={
                    "active": True,
                    "operator_id": str(operator_id),
                    "email": "oleh@company.com",
                    "status": "ACTIVE",
                    "authorities": ["PROMPTS"],
                },
            )
        return httpx.Response(503, text="down")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://java-service:8080", transport=transport) as http_client:
        auth_client = CoreAuthClient(Settings(), client=http_client)
        res = await auth_client.introspect("good")
        assert res.active is True
        assert res.operator_id == operator_id
        assert res.authorities == ["PROMPTS"]

        with pytest.raises(DashboardApiError) as exc_info:
            await auth_client.introspect("fail")
        assert exc_info.value.http_status == 500
        assert exc_info.value.error_code == DashboardErrorCode.INTERNAL_ERROR


async def test_grounded_answer_returns_prompt_version_id() -> None:
    prompt_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
    db = MagicMock()
    db.query_with_auto_merge = AsyncMock(
        return_value=[
            {
                "document_id": "doc-1",
                "chunk_ids": ["chunk-1"],
                "chunks": ["Internet compensation is available on SharePoint."],
                "score": 0.92,
            }
        ]
    )
    db.get_latest_prompt_version = AsyncMock(
        return_value=PromptVersion(
            id=prompt_id,
            prompt_text="Answer strictly.",
            metadata={"model": "gpt-4o-mini"},
        )
    )
    vectorizer = MagicMock()
    vectorizer.vectorize_query = MagicMock(return_value=[0.1, 0.2])
    llm = MagicMock()
    llm.generate_json = AsyncMock(
        return_value=json.dumps(
            {
                "answer": "Submit the form on SharePoint.",
                "confidence_score": 0.9,
                "insufficient_evidence": False,
                "cited_chunk_ids": ["chunk-1"],
            }
        )
    )

    service = GroundedAnswerService(db=db, vectorizer=vectorizer, llm=llm, settings=Settings())
    req = RagGenerationRequest(
        query_id="q-1",
        query="How to get internet compensation?",
        user_permissions=UserPermissions(allowed_groups=["all-employees"]),
    )

    response = await service.generate(req)
    assert response.status == RagGenerationStatus.ANSWER_GENERATED
    assert response.prompt_version_id == prompt_id
