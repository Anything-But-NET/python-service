import logging
import time
from typing import Any

import httpx

from python_service.config import Settings

logger = logging.getLogger(__name__)

_PLACEHOLDER_API_KEYS = frozenset({
    "",
    "sk-or-v1-xxxx",
    "sk-xxxx",
    "your-api-key",
    "changeme",
    "placeholder",
})


class LLMProviderError(RuntimeError):
    """Raised when the configured OpenAI-compatible provider cannot generate."""


class OpenAICompatibleClient:
    """Small async adapter for OpenAI Chat Completions-compatible providers."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        health_cache_ttl_seconds: float = 60.0,
    ) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key.get_secret_value()
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_timeout_seconds),
        )
        self._health_cache_ttl_seconds = health_cache_ttl_seconds
        self._cached_remote_health: tuple[float, bool] | None = None

    def is_configured(self) -> bool:
        key = (self.api_key or "").strip()
        if not key or key.lower() in _PLACEHOLDER_API_KEYS:
            return False
        return bool(self.base_url and self.base_url.strip() and self.model and self.model.strip())

    async def check_health(self, verify_remote: bool = False, timeout: float = 1.5) -> bool:
        if not self.is_configured():
            return False
        if not verify_remote:
            return True

        now = time.monotonic()
        if (
            self._cached_remote_health is not None
            and (now - self._cached_remote_health[0]) < self._health_cache_ttl_seconds
        ):
            return self._cached_remote_health[1]

        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=timeout,
            )
            ok = response.status_code == 200
        except httpx.HTTPError:
            ok = False

        self._cached_remote_health = (now, ok)
        return ok

    async def generate_json(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
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

        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        payload.update(valid_kwargs)

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
