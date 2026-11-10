import json

import httpx
import pytest

from python_service.config import Settings
from python_service.services.llm_client import LLMProviderError, OpenAICompatibleClient


@pytest.mark.asyncio
async def test_openai_compatible_client_sends_json_chat_completion_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = request.read()
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"answer":"ok"}'}}],
        })

    settings = Settings(
        _env_file=None,
        llm_base_url="https://llm.example/v1/",
        llm_api_key="test-secret",
        llm_model="test-model",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        llm = OpenAICompatibleClient(settings, client=http_client)
        result = await llm.generate_json([{"role": "system", "content": "safe"}])

    assert result == '{"answer":"ok"}'
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-secret"
    payload = json.loads(captured["body"])
    assert payload["model"] == "test-model"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0


@pytest.mark.asyncio
async def test_openai_compatible_client_raises_provider_error_for_http_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    settings = Settings(_env_file=None, llm_base_url="https://llm.example/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        llm = OpenAICompatibleClient(settings, client=http_client)
        with pytest.raises(LLMProviderError):
            await llm.generate_json([{"role": "system", "content": "safe"}])
