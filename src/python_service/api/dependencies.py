from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Header, Request

from python_service.api.errors import DashboardApiError
from python_service.api.schemas import DashboardErrorCode, TokenIntrospectResponse
from python_service.db.service import RagDbService
from python_service.services.auth_client import CoreAuthClient
from python_service.services.evaluation_service import EvaluationService
from python_service.services.grounded_answer_service import GroundedAnswerService
from python_service.services.parse_service import MarkdownParserService
from python_service.services.transform_service import VectorizationService


def get_db_service(request: Request) -> RagDbService:
    return request.app.state.db_service


def get_vectorizer_service(request: Request) -> VectorizationService:
    return request.app.state.vectorizer_service


def get_parser_service(request: Request) -> MarkdownParserService:
    return request.app.state.parser_service


def get_grounded_answer_service(request: Request) -> GroundedAnswerService:
    return request.app.state.grounded_answer_service


def get_evaluation_service(request: Request) -> EvaluationService:
    return request.app.state.evaluation_service


def get_auth_client(request: Request) -> CoreAuthClient:
    return request.app.state.auth_client


def require_dashboard_authority(
    required_authority: str,
) -> Callable[..., Awaitable[TokenIntrospectResponse]]:
    async def _dependency(
        auth_client: Annotated[CoreAuthClient, Depends(get_auth_client)],
        authorization: Annotated[str | None, Header()] = None,
    ) -> TokenIntrospectResponse:
        if not authorization or not authorization.startswith("Bearer "):
            raise DashboardApiError(
                http_status=401,
                error_code=DashboardErrorCode.UNAUTHORIZED,
                message="Missing or expired Bearer access token.",
            )
        token = authorization[len("Bearer "):].strip()
        if not token:
            raise DashboardApiError(
                http_status=401,
                error_code=DashboardErrorCode.UNAUTHORIZED,
                message="Missing or expired Bearer access token.",
            )

        introspected = await auth_client.introspect(token)
        if not introspected.active:
            raise DashboardApiError(
                http_status=401,
                error_code=DashboardErrorCode.UNAUTHORIZED,
                message="Missing or expired Bearer access token.",
            )
        if required_authority not in introspected.authorities:
            raise DashboardApiError(
                http_status=403,
                error_code=DashboardErrorCode.FORBIDDEN,
                message=f"Missing required authority '{required_authority}'.",
            )
        return introspected

    return _dependency
