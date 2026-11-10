import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from python_service.api.errors import GenerationServiceError
from python_service.api.exception_handlers import install_generation_exception_handlers
from python_service.api.schemas import (
    GenerationErrorStatus,
    RagGenerationRequest,
)
from python_service.config import Settings


def make_client() -> TestClient:
    app = FastAPI()
    install_generation_exception_handlers(app)

    @app.post("/api/v1/rag/generate")
    async def validate_payload(payload: RagGenerationRequest):
        if payload.query == "raise-service-error":
            raise GenerationServiceError(
                GenerationErrorStatus.LLM_PROVIDER_ERROR,
                "LLM provider is unavailable",
                query_id=payload.query_id,
            )
        return {"query_id": payload.query_id}

    return TestClient(app, raise_server_exceptions=False)


def test_schema_validation_returns_generation_error_422():
    client = make_client()
    response = client.post(
        "/api/v1/rag/generate",
        json={
            "query_id": "query-422",
            "query": "x" * 4001,
            "user_permissions": {"allowed_groups": ["employees"]},
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["query_id"] == "query-422"
    assert body["status"] == "VALIDATION_ERROR"
    assert body["message"] == "Request validation failed"
    assert isinstance(body["timestamp"], int)
    assert body["description"]


def test_malformed_json_returns_generation_error_400():
    client = make_client()
    response = client.post(
        "/api/v1/rag/generate",
        content="{not-json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["query_id"] is None
    assert body["status"] == "VALIDATION_ERROR"
    assert body["message"] == "Malformed request payload"
    assert body["timestamp"] > 0


def test_settings_reject_misaligned_relevance_and_confidence_thresholds():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            rag_min_relevance_score=0.45,
            rag_min_confidence=0.6,
        )


def test_service_failure_returns_generation_error_503():
    client = make_client()
    response = client.post(
        "/api/v1/rag/generate",
        json={
            "query_id": "query-503",
            "query": "raise-service-error",
            "user_permissions": {"allowed_groups": ["employees"]},
        },
    )

    assert response.status_code == 503
    body = response.json()
    assert body["query_id"] == "query-503"
    assert body["status"] == "LLM_PROVIDER_ERROR"
    assert body["message"] == "LLM provider is unavailable"
    assert isinstance(body["timestamp"], int)
