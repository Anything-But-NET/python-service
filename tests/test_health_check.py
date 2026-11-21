import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from python_service.config import Settings
from python_service.kafka.consumer import IngestionKafkaWorker
from python_service.main import create_app
from python_service.services.llm_client import OpenAICompatibleClient
from python_service.services.transform_service import VectorizationService


def _build_test_app(
    *,
    db_ok: bool = True,
    vectorizer_ok: bool = True,
    kafka_ok: bool = True,
    kafka_task_done: bool = False,
    llm_ok: bool = True,
):
    app = create_app()

    db = MagicMock()
    db.check_health = AsyncMock(return_value=db_ok)
    app.state.db_service = db

    vectorizer = MagicMock()
    vectorizer.check_health = MagicMock(return_value=vectorizer_ok)
    app.state.vectorizer_service = vectorizer

    if kafka_ok is None:
        app.state.kafka_worker = None
    else:
        kafka_worker = MagicMock()
        kafka_worker.check_health = MagicMock(return_value=kafka_ok)
        app.state.kafka_worker = kafka_worker

    task = MagicMock()
    task.done.return_value = kafka_task_done
    app.state.kafka_task = task

    llm = MagicMock()
    llm.check_health = AsyncMock(return_value=llm_ok)
    app.state.llm_client = llm

    return app


def test_health_endpoint_returns_up_when_all_dependencies_healthy():
    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "UP",
        "components": {
            "db": {"status": "UP"},
            "vectorizer": {"status": "UP"},
            "kafka": {"status": "UP"},
            "llm": {"status": "UP"},
        },
    }


def test_health_endpoint_returns_degraded_when_only_kafka_is_down():
    app = _build_test_app(kafka_ok=False)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "DEGRADED"
    assert body["components"]["db"]["status"] == "UP"
    assert body["components"]["vectorizer"]["status"] == "UP"
    assert body["components"]["llm"]["status"] == "UP"
    assert body["components"]["kafka"]["status"] == "DOWN"
    assert "reason" in body["components"]["kafka"]["details"]


def test_health_endpoint_returns_degraded_when_kafka_task_stopped():
    app = _build_test_app(kafka_ok=True, kafka_task_done=True)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "DEGRADED"
    assert body["components"]["kafka"]["status"] == "DOWN"


@pytest.mark.parametrize(
    "failing_field",
    ["db_ok", "vectorizer_ok", "llm_ok"],
)
def test_health_endpoint_returns_503_down_when_critical_dependency_fails(failing_field):
    kwargs = {failing_field: False}
    app = _build_test_app(**kwargs)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "DOWN"


@pytest.mark.asyncio
async def test_health_endpoint_never_exposes_secrets_on_failure():
    app = create_app()
    secret_db_url = "postgresql://postgres:SuperSecretDbPass123@db:5432/copilot_db"
    secret_api_key = "sk-live-SuperSecretApiKey987654321"

    db = MagicMock()
    db.check_health = AsyncMock(side_effect=RuntimeError(f"Failed connecting to {secret_db_url}"))
    app.state.db_service = db

    vectorizer = MagicMock()
    vectorizer.check_health = MagicMock(return_value=True)
    app.state.vectorizer_service = vectorizer

    app.state.kafka_worker = None
    app.state.kafka_task = None
    app.state.settings = Settings(DB_URL=secret_db_url, LLM_API_KEY=secret_api_key)
    llm_client = OpenAICompatibleClient(app.state.settings)
    app.state.llm_client = llm_client

    try:
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 503
        raw_text = response.text
        assert "SuperSecretDbPass123" not in raw_text
        assert secret_db_url not in raw_text
        assert secret_api_key not in raw_text
        assert "SuperSecretApiKey987654321" not in raw_text
    finally:
        await llm_client.aclose()


@pytest.mark.asyncio
async def test_health_endpoint_uses_cached_remote_llm_check_when_enabled():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        settings = Settings(
            LLM_API_KEY="sk-real-key",
            LLM_BASE_URL="https://api.example.com/v1",
            LLM_HEALTH_VERIFY_REMOTE=True,
        )
        app = _build_test_app()
        app.state.settings = settings
        app.state.llm_client = OpenAICompatibleClient(settings, client=http_client, health_cache_ttl_seconds=60.0)

        client = TestClient(app)
        first = client.get("/health")
        second = client.get("/health")

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["components"]["llm"]["status"] == "UP"
        assert call_count == 1


@pytest.mark.asyncio
async def test_llm_client_health_check_rejects_empty_and_placeholder_keys():
    empty_client = OpenAICompatibleClient(Settings(LLM_API_KEY=""))
    assert await empty_client.check_health() is False

    placeholder_client = OpenAICompatibleClient(Settings(LLM_API_KEY="sk-or-v1-xxxx"))
    assert await placeholder_client.check_health() is False

    valid_client = OpenAICompatibleClient(Settings(LLM_API_KEY="sk-or-v1-real-configured-key"))
    assert await valid_client.check_health() is True


@pytest.mark.asyncio
async def test_llm_client_cached_remote_health_check_avoids_spam():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        llm = OpenAICompatibleClient(
            Settings(LLM_API_KEY="sk-real-key", LLM_BASE_URL="https://api.example.com/v1"),
            client=http_client,
            health_cache_ttl_seconds=60.0,
        )
        assert await llm.check_health(verify_remote=True) is True
        assert await llm.check_health(verify_remote=True) is True
        assert call_count == 1


def test_vectorizer_service_check_health():
    channel = MagicMock()
    service = VectorizationService(grpc_target="localhost:50051", channel=channel)

    ready_future = MagicMock()
    ready_future.result.return_value = None
    with patch("python_service.services.transform_service.grpc.channel_ready_future", return_value=ready_future):
        assert service.check_health(timeout=1.0) is True

    failing_future = MagicMock()
    failing_future.result.side_effect = TimeoutError("gRPC channel timeout")
    with patch("python_service.services.transform_service.grpc.channel_ready_future", return_value=failing_future):
        assert service.check_health(timeout=1.0) is False


def test_kafka_worker_check_health():
    with (
        patch("python_service.kafka.consumer.Consumer"),
        patch("python_service.kafka.consumer.Producer") as mock_producer_cls,
    ):
        mock_producer = MagicMock()
        metadata = MagicMock()
        metadata.brokers = {1: MagicMock()}
        mock_producer.list_topics.return_value = metadata
        mock_producer_cls.return_value = mock_producer

        worker = IngestionKafkaWorker(
            parser=MagicMock(),
            vectorizer=MagicMock(),
            db=AsyncMock(),
        )
        assert worker.check_health(timeout=1.0) is True

        mock_producer.list_topics.side_effect = RuntimeError("broker down")
        assert worker.check_health(timeout=1.0) is False
