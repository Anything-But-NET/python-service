import json
import logging
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from python_service.api.errors import GenerationServiceError
from python_service.api.schemas import GenerationError, GenerationErrorStatus

logger = logging.getLogger(__name__)
_GENERATE_PATH = "/api/v1/rag/generate"


def _query_id_from_body(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    value = body.get("query_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def install_generation_exception_handlers(app: FastAPI) -> None:
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        if request.url.path != _GENERATE_PATH:
            return await request_validation_exception_handler(request, exc)

        errors = exc.errors()
        malformed_json = any(error.get("type") == "json_invalid" for error in errors)
        body: Any = getattr(exc, "body", None)
        if body is None:
            try:
                body = await request.json()
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = None

        if malformed_json:
            message = "Malformed request payload"
            description = "JSON body could not be parsed."
            http_status = 400
        else:
            message = "Request validation failed"
            details = [
                f"{'.'.join(str(part) for part in error.get('loc', ()) if part != 'body')}: "
                f"{error.get('msg', 'Invalid value')}"
                for error in errors
            ]
            description = "; ".join(details) or None
            http_status = 422

        error = GenerationError(
            query_id=_query_id_from_body(body),
            status=GenerationErrorStatus.VALIDATION_ERROR,
            message=message,
            description=description,
            timestamp=int(time.time()),
        )
        return JSONResponse(status_code=http_status, content=error.model_dump(mode="json"))

    async def generation_service_error_handler(
        request: Request,
        exc: GenerationServiceError,
    ) -> JSONResponse:
        error = GenerationError(
            query_id=exc.query_id,
            status=exc.status,
            message=exc.message,
            description=exc.description,
            timestamp=int(time.time()),
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=error.model_dump(mode="json"),
        )

    async def unexpected_error_handler(request: Request, exc: Exception):
        if request.url.path != _GENERATE_PATH:
            logger.error(
                "Unhandled API error",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return PlainTextResponse("Internal Server Error", status_code=500)

        logger.error(
            "Unexpected grounded RAG failure",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        error = GenerationError(
            query_id=None,
            status=GenerationErrorStatus.INTERNAL_ERROR,
            message="Unexpected failure during RAG pipeline execution",
            description=None,
            timestamp=int(time.time()),
        )
        return JSONResponse(status_code=500, content=error.model_dump(mode="json"))

    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(GenerationServiceError, generation_service_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
