from __future__ import annotations

import logging

import httpx

from python_service.api.errors import DashboardApiError
from python_service.api.schemas import DashboardErrorCode, TokenIntrospectResponse
from python_service.config import Settings

logger = logging.getLogger(__name__)


class CoreAuthClient:
    """Calls java-service /api/v1/internal/auth/introspect to verify dashboard operator JWTs."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.core_api_url.rstrip("/"),
            timeout=settings.core_introspect_timeout_seconds,
        )

    async def introspect(self, token: str) -> TokenIntrospectResponse:
        try:
            response = await self._client.post(
                "/api/v1/internal/auth/introspect",
                json={"token": token},
            )
            response.raise_for_status()
            return TokenIntrospectResponse.model_validate(response.json())
        except httpx.HTTPError as exc:
            logger.warning("Core token introspection request failed: %s", exc)
            raise DashboardApiError(
                http_status=500,
                error_code=DashboardErrorCode.INTERNAL_ERROR,
                message="Authentication service is unavailable.",
            ) from exc
        except Exception as exc:
            logger.warning("Core token introspection returned invalid payload: %s", exc)
            raise DashboardApiError(
                http_status=500,
                error_code=DashboardErrorCode.INTERNAL_ERROR,
                message="Authentication service returned an invalid response.",
            ) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
