import logging
from typing import Any

import httpx

from python_service.config import Settings

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Raised when the configured OpenAI-compatible provider cannot generate."""


class OpenAICompatibleClient:
    """Small async adapter for OpenAI Chat Completions-compatible providers."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key.get_secret_value()
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_timeout_seconds),
        )

    async def generate_json(self, messages: list[dict[str, str]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("LLM returned an empty response")
            return content
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("LLM provider request failed: %s", exc)
            raise LLMProviderError("LLM provider request failed") from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()
