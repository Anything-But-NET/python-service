import asyncio
import math
from typing import Annotated, Any
import uuid

from fastapi import APIRouter, Depends, Query, Request, Response, status

from python_service.api.dependencies import (
    get_db_service,
    get_evaluation_service,
    get_grounded_answer_service,
    get_vectorizer_service,
    require_dashboard_authority,
)
from python_service.api.errors import DashboardApiError
from python_service.api.schemas import (
    ComponentHealth,
    DashboardErrorCode,
    DashboardErrorResponse,
    EvaluationRunCreateRequest,
    EvaluationRunResponse,
    GenerationError,
    HealthComponents,
    HealthResponse,
    HealthStatus,
    PromptVersionDetail,
    PromptVersionsPage,
    PromptVersionSummary,
    RagCitation,
    RagGenerationRequest,
    RagGenerationResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from python_service.config import get_settings
from python_service.db.service import RagDbService
from python_service.services.evaluation_service import EvaluationService
from python_service.services.grounded_answer_service import GroundedAnswerService
from python_service.services.llm_client import OpenAICompatibleClient
from python_service.services.transform_service import VectorizationService

router = APIRouter()

HEALTH_CHECK_TIMEOUT_SECONDS = 1.5


def _to_search_result(result: dict) -> SearchResultItem:
    citations = [
        RagCitation(
            document_id=result["document_id"],
            chunk_id=chunk_id,
            score=result["score"],
        )
        for chunk_id in result["chunk_ids"]
    ]
    return SearchResultItem(**result, citations=citations)


def _resolve_state_or_override(request: Request, dep_fn: Any, state_attr: str) -> Any:
    override = request.app.dependency_overrides.get(dep_fn)
    if override is not None:
        return override()
    return getattr(request.app.state, state_attr, None)


async def _check_db_component(db: Any) -> ComponentHealth:
    if db is None:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Database service not initialized"})
    try:
        is_up = await asyncio.wait_for(db.check_health(), timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        if is_up:
            return ComponentHealth(status=HealthStatus.UP)
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Database connectivity check failed"})
    except TimeoutError:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Database health check timed out"})
    except Exception:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Database connectivity check failed"})


async def _check_vectorizer_component(vectorizer: Any) -> ComponentHealth:
    if vectorizer is None:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Vectorizer service not initialized"})
    try:
        is_up = await asyncio.wait_for(
            asyncio.to_thread(vectorizer.check_health, HEALTH_CHECK_TIMEOUT_SECONDS),
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS + 0.5,
        )
        if is_up:
            return ComponentHealth(status=HealthStatus.UP)
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Vectorizer gRPC channel not ready"})
    except TimeoutError:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Vectorizer health check timed out"})
    except Exception:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Vectorizer gRPC channel not ready"})


async def _check_kafka_component(kafka_worker: Any, kafka_task: asyncio.Task[None] | None) -> ComponentHealth:
    if kafka_worker is None:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Kafka ingestion worker not running"})
    if kafka_task is not None and kafka_task.done():
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Kafka ingestion background task stopped"})
    try:
        is_up = await asyncio.wait_for(
            asyncio.to_thread(kafka_worker.check_health, HEALTH_CHECK_TIMEOUT_SECONDS),
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS + 0.5,
        )
        if is_up:
            return ComponentHealth(status=HealthStatus.UP)
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Kafka broker unreachable"})
    except TimeoutError:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Kafka health check timed out"})
    except Exception:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "Kafka broker unreachable"})


async def _check_llm_component(llm_client: Any, settings: Any) -> ComponentHealth:
    resolved_settings = settings or get_settings()
    verify_remote = bool(getattr(resolved_settings, "llm_health_verify_remote", False))
    created_client = False
    client = llm_client
    try:
        if client is None:
            client = OpenAICompatibleClient(resolved_settings)
            created_client = True
        is_up = await asyncio.wait_for(
            client.check_health(verify_remote=verify_remote, timeout=HEALTH_CHECK_TIMEOUT_SECONDS),
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS + 0.5,
        )
        if is_up:
            return ComponentHealth(status=HealthStatus.UP)
        reason = "LLM provider check failed or LLM_API_KEY is not configured" if verify_remote else "LLM_API_KEY is not configured"
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": reason})
    except TimeoutError:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "LLM health check timed out"})
    except Exception:
        return ComponentHealth(status=HealthStatus.DOWN, details={"reason": "LLM configuration check failed"})
    finally:
        if created_client and client is not None and hasattr(client, "aclose"):
            await client.aclose()


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    responses={
        200: {"model": HealthResponse, "description": "Service is operational (UP or DEGRADED)."},
        503: {"model": HealthResponse, "description": "Service is unavailable (DOWN)."},
    },
    tags=["system"],
)
async def health_check(request: Request, response: Response) -> HealthResponse:
    db = _resolve_state_or_override(request, get_db_service, "db_service")
    vectorizer = _resolve_state_or_override(request, get_vectorizer_service, "vectorizer_service")
    kafka_worker = getattr(request.app.state, "kafka_worker", None)
    kafka_task = getattr(request.app.state, "kafka_task", None)
    llm_client = getattr(request.app.state, "llm_client", None)
    settings = getattr(request.app.state, "settings", None)

    db_health, vectorizer_health, kafka_health, llm_health = await asyncio.gather(
        _check_db_component(db),
        _check_vectorizer_component(vectorizer),
        _check_kafka_component(kafka_worker, kafka_task),
        _check_llm_component(llm_client, settings),
    )

    critical_down = any(
        comp.status == HealthStatus.DOWN
        for comp in (db_health, vectorizer_health, llm_health)
    )
    if critical_down:
        overall_status = HealthStatus.DOWN
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif kafka_health.status == HealthStatus.DOWN:
        overall_status = HealthStatus.DEGRADED
        response.status_code = status.HTTP_200_OK
    else:
        overall_status = HealthStatus.UP
        response.status_code = status.HTTP_200_OK

    return HealthResponse(
        status=overall_status,
        components=HealthComponents(
            db=db_health,
            vectorizer=vectorizer_health,
            kafka=kafka_health,
            llm=llm_health,
        ),
    )


@router.post("/search", response_model=SearchResponse, tags=["rag"])
async def search_documents(
    payload: SearchRequest,
    db: Annotated[RagDbService, Depends(get_db_service)],
    vectorizer: Annotated[VectorizationService, Depends(get_vectorizer_service)],
) -> SearchResponse:
    query_vec = await asyncio.to_thread(vectorizer.vectorize_query, payload.query)
    results = await db.query_with_auto_merge(
        vector=query_vec,
        allowed_groups=payload.user_permissions.allowed_groups,
        limit=payload.limit,
        threshold=payload.threshold,
    )
    return SearchResponse(results=[_to_search_result(r) for r in results])


@router.post(
    "/api/v1/rag/generate",
    response_model=RagGenerationResponse,
    responses={
        400: {"model": GenerationError, "description": "Malformed request payload."},
        422: {"model": GenerationError, "description": "Request schema validation failed."},
        500: {"model": GenerationError, "description": "Unexpected RAG pipeline failure."},
        503: {"model": GenerationError, "description": "Vector DB, vectorizer, or LLM provider unavailable."},
    },
    tags=["rag"],
)
async def generate_grounded_answer(
    payload: RagGenerationRequest,
    service: Annotated[GroundedAnswerService, Depends(get_grounded_answer_service)],
) -> RagGenerationResponse:
    return await service.generate(payload)


@router.post(
    "/api/v1/rag/evaluations/runs",
    status_code=status.HTTP_201_CREATED,
    response_model=EvaluationRunResponse,
    responses={
        400: {"model": GenerationError, "description": "Bad Request."},
        404: {"model": GenerationError, "description": "Prompt version or Golden Set dataset not found."},
        422: {"model": GenerationError, "description": "Unprocessable Entity."},
        500: {"model": GenerationError, "description": "Internal Server Error."},
        503: {"model": GenerationError, "description": "Service Unavailable."},
    },
    tags=["RagEvaluation"],
)
async def run_quality_evaluation(
    payload: EvaluationRunCreateRequest,
    service: Annotated[EvaluationService, Depends(get_evaluation_service)],
) -> EvaluationRunResponse:
    return await service.run_evaluation(payload)


@router.get(
    "/api/v1/rag/evaluations/runs/{run_id}",
    response_model=EvaluationRunResponse,
    responses={
        404: {"model": GenerationError, "description": "Evaluation run not found."},
        422: {"model": GenerationError, "description": "Unprocessable Entity."},
        500: {"model": GenerationError, "description": "Internal Server Error."},
    },
    tags=["RagEvaluation"],
)
async def get_evaluation_run(
    run_id: uuid.UUID,
    service: Annotated[EvaluationService, Depends(get_evaluation_service)],
) -> EvaluationRunResponse:
    return await service.get_run(str(run_id))


@router.get(
    "/api/v1/dash/prompts",
    response_model=PromptVersionsPage,
    dependencies=[Depends(require_dashboard_authority("PROMPTS"))],
    responses={
        400: {"model": DashboardErrorResponse, "description": "Bad Request."},
        401: {"model": DashboardErrorResponse, "description": "Unauthorized."},
        403: {"model": DashboardErrorResponse, "description": "Forbidden."},
    },
    tags=["Dashboard: Prompts (RAG)"],
)
async def list_prompt_versions(
    db: Annotated[RagDbService, Depends(get_db_service)],
    page: int = Query(default=0, ge=0),
    size: int = Query(default=20, ge=1, le=100),
) -> PromptVersionsPage:
    prompts, total_elements = await db.list_prompt_versions(page=page, size=size)
    total_pages = math.ceil(total_elements / size) if total_elements > 0 else 0
    return PromptVersionsPage(
        items=[
            PromptVersionSummary(id=prompt.id, created_at=prompt.created_at)
            for prompt in prompts
        ],
        page=page,
        size=size,
        total_elements=total_elements,
        total_pages=total_pages,
    )


@router.get(
    "/api/v1/dash/prompts/{prompt_version_id}",
    response_model=PromptVersionDetail,
    dependencies=[Depends(require_dashboard_authority("PROMPTS"))],
    responses={
        400: {"model": DashboardErrorResponse, "description": "Bad Request."},
        401: {"model": DashboardErrorResponse, "description": "Unauthorized."},
        403: {"model": DashboardErrorResponse, "description": "Forbidden."},
        404: {"model": DashboardErrorResponse, "description": "Prompt version not found."},
    },
    tags=["Dashboard: Prompts (RAG)"],
)
async def get_prompt_version_by_id(
    prompt_version_id: uuid.UUID,
    db: Annotated[RagDbService, Depends(get_db_service)],
) -> PromptVersionDetail:
    prompt = await db.get_prompt_version(prompt_version_id)
    if prompt is None:
        raise DashboardApiError(
            http_status=404,
            error_code=DashboardErrorCode.NOT_FOUND,
            message=f"Prompt version '{prompt_version_id}' not found.",
        )
    return PromptVersionDetail(
        id=prompt.id,
        prompt_text=prompt.prompt_text,
        metadata=prompt.metadata,
        created_at=prompt.created_at,
    )

