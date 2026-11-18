import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from python_service.api.exception_handlers import install_generation_exception_handlers
from python_service.api.routes import router
from python_service.api.schemas import (
    EvaluationDatasetConfig,
    EvaluationRunCreateRequest,
    EvaluationRunStatus,
    ExpectedOutcomeType,
    QualityDimension,
    QualityThresholdsConfig,
    RagCitation,
    RagGenerationResponse,
    RagGenerationStatus,
    ThresholdDecisionStatus,
)
from python_service.models.domain import EvaluationRun, PromptVersion
from python_service.services.evaluation_service import (
    EvaluationService,
    check_citation_correctness,
    check_factual_grounding,
    check_relevance,
    check_safe_no_answer,
)


def test_check_relevance_passes_when_expected_answer_matches():
    res = check_relevance(
        query="What tool is used for containerization in the CI/CD pipeline?",
        expected_answer="Docker",
        actual_answer="Docker is used for containerization in the CI/CD pipeline.",
        expected_outcome=ExpectedOutcomeType.ANSWER_EXPECTED,
        threshold=0.75,
    )
    assert res.applicable is True
    assert res.passed is True
    assert res.score >= 0.75


def test_check_relevance_fails_on_abstention_when_answer_expected():
    res = check_relevance(
        query="What tool is used for containerization?",
        expected_answer="Docker",
        actual_answer="I couldn't find information about this in the knowledge base.",
        expected_outcome=ExpectedOutcomeType.ANSWER_EXPECTED,
        threshold=0.75,
    )
    assert res.applicable is True
    assert res.passed is False
    assert res.score == 0.0


def test_check_factual_grounding_detects_unsupported_claims():
    passages = [
        {
            "document_id": "eng-backend-deployment-pipeline-architecture",
            "chunk_id": "123e4567-e89b-12d3-a456-426614174000",
            "score": 0.91,
            "text": "Stage 3 runs SonarQube for static code analysis and Trivy for vulnerability scanning.",
        }
    ]
    res_supported = check_factual_grounding(
        actual_answer="Stage 3 uses SonarQube for static code analysis.",
        retrieved_passages=passages,
        expected_outcome=ExpectedOutcomeType.ANSWER_EXPECTED,
        threshold=0.85,
    )
    assert res_supported.passed is True
    assert res_supported.score == 1.0
    assert res_supported.details["unsupported_claims"] == []

    res_unsupported = check_factual_grounding(
        actual_answer="Stage 3 uses SonarQube. It also deploys directly to Azure Functions.",
        retrieved_passages=passages,
        expected_outcome=ExpectedOutcomeType.ANSWER_EXPECTED,
        threshold=0.85,
    )
    assert res_unsupported.passed is False
    assert res_unsupported.score == 0.5
    assert len(res_unsupported.details["unsupported_claims"]) == 1


def test_check_citation_correctness_verifies_source_and_support():
    chunk_id = "123e4567-e89b-12d3-a456-426614174000"
    passages = [
        {
            "document_id": "eng-backend-deployment-pipeline-architecture",
            "chunk_id": chunk_id,
            "score": 0.95,
            "text": "Artifacts are stored in Amazon ECR in us-east-1.",
        }
    ]
    good_citations = [
        RagCitation(
            document_id="eng-backend-deployment-pipeline-architecture",
            chunk_id=chunk_id,
            score=0.95,
        )
    ]
    res_good = check_citation_correctness(
        actual_answer="Artifacts are stored in Amazon ECR.",
        actual_citations=good_citations,
        retrieved_passages=passages,
        expected_document_ids=["eng-backend-deployment-pipeline-architecture"],
        expected_outcome=ExpectedOutcomeType.ANSWER_EXPECTED,
        threshold=1.0,
    )
    assert res_good.passed is True
    assert res_good.score == 1.0

    wrong_doc_citations = [
        RagCitation(
            document_id="wrong-document-id",
            chunk_id=chunk_id,
            score=0.95,
        )
    ]
    res_wrong = check_citation_correctness(
        actual_answer="Artifacts are stored in Amazon ECR.",
        actual_citations=wrong_doc_citations,
        retrieved_passages=passages,
        expected_document_ids=["eng-backend-deployment-pipeline-architecture"],
        expected_outcome=ExpectedOutcomeType.ANSWER_EXPECTED,
        threshold=1.0,
    )
    assert res_wrong.passed is False
    assert res_wrong.details["resolves_to_expected_source"] is False
    assert res_wrong.details["unexpected_document_ids"] == ["wrong-document-id"]


def test_check_safe_no_answer_distinguishes_abstention_and_hallucination():
    res_good_abstention = check_safe_no_answer(
        expected_outcome=ExpectedOutcomeType.INSUFFICIENT_EVIDENCE,
        actual_status=RagGenerationStatus.EMPTY_RETRIEVAL,
        actual_answer=None,
        actual_citations=[],
    )
    assert res_good_abstention.applicable is True
    assert res_good_abstention.passed is True
    assert res_good_abstention.score == 1.0

    res_hallucination = check_safe_no_answer(
        expected_outcome=ExpectedOutcomeType.INSUFFICIENT_EVIDENCE,
        actual_status=RagGenerationStatus.ANSWER_GENERATED,
        actual_answer="The guest Wi-Fi password is Password123.",
        actual_citations=[
            RagCitation(
                document_id="doc-1",
                chunk_id="123e4567-e89b-12d3-a456-426614174000",
                score=0.8,
            )
        ],
    )
    assert res_hallucination.applicable is True
    assert res_hallucination.passed is False
    assert res_hallucination.score == 0.0


@pytest.mark.asyncio
async def test_evaluation_service_runs_on_60_golden_set_cases_and_persists_report():
    db = MagicMock()
    prompt_id = uuid.uuid4()
    run_id = uuid.uuid4()
    db.get_prompt_version = AsyncMock(
        return_value=PromptVersion(id=prompt_id, prompt_text="system prompt", metadata={})
    )

    async def _fake_create_run(prompt_version_id, status, metadata):
        return EvaluationRun(
            id=run_id,
            prompt_version_id=prompt_version_id,
            status=status,
            metadata=metadata,
            created_at=datetime.now(UTC),
        )

    db.create_evaluation_run = AsyncMock(side_effect=_fake_create_run)

    # Load actual 60-case Golden Set to build realistic mocked RAG responses
    grounded_service = MagicMock()
    grounded_service.settings.rag_retrieval_limit = 20
    grounded_service.settings.rag_auto_merge_threshold = 0.5
    grounded_service.vectorizer.vectorize_query = MagicMock(return_value=[0.1] * 1024)

    eval_service = EvaluationService(db=db, grounded_answer_service=grounded_service)
    golden_data = eval_service.load_golden_set(
        EvaluationDatasetConfig(dataset_id="mvp-golden-set", dataset_version="v1.0.0")
    )
    assert len(golden_data["cases"]) == 60

    case_by_query = {c["query"]: c for c in golden_data["cases"]}
    chunk_uuid = "123e4567-e89b-12d3-a456-426614174000"

    async def _fake_query_with_auto_merge(vector, allowed_groups, limit, threshold):
        return []

    db.query_with_auto_merge = AsyncMock(side_effect=_fake_query_with_auto_merge)

    def _fake_prepare_passages(results):
        return [
            {
                "document_id": "eng-backend-deployment-pipeline-architecture",
                "chunk_id": chunk_uuid,
                "score": 0.92,
                "text": " ".join(
                    c["expected_answer"]
                    for c in golden_data["cases"]
                    if c["expected_outcome"] == "ANSWER_EXPECTED"
                ),
            }
        ]

    grounded_service._prepare_passages = MagicMock(side_effect=_fake_prepare_passages)

    async def _fake_generate(req):
        case = case_by_query[req.query]
        if case["expected_outcome"] == "INSUFFICIENT_EVIDENCE":
            return RagGenerationResponse(
                query_id=req.query_id,
                status=RagGenerationStatus.EMPTY_RETRIEVAL,
                answer=None,
                confidence_score=0.0,
                citations=[],
            )
        return RagGenerationResponse(
            query_id=req.query_id,
            status=RagGenerationStatus.ANSWER_GENERATED,
            answer=case["expected_answer"],
            confidence_score=0.92,
            citations=[
                RagCitation(
                    document_id="eng-backend-deployment-pipeline-architecture",
                    chunk_id=chunk_uuid,
                    score=0.92,
                )
            ],
        )

    grounded_service.generate = AsyncMock(side_effect=_fake_generate)

    request = EvaluationRunCreateRequest(
        prompt_version_id=prompt_id,
        dataset=EvaluationDatasetConfig(
            dataset_id="mvp-golden-set",
            dataset_version="v1.0.0",
        ),
        thresholds=QualityThresholdsConfig(
            decision_status=ThresholdDecisionStatus.PROPOSED,
            relevance_min_score=0.75,
            factual_grounding_min_score=0.85,
            citation_correctness_min_score=1.0,
            target_pass_rate=0.85,
        ),
    )

    response = await eval_service.run_evaluation(request)
    assert response.id == run_id
    assert response.status == EvaluationRunStatus.COMPLETED
    assert response.metadata.total_cases == 60
    assert response.metadata.passed_cases == 60
    assert response.metadata.failed_cases == 0
    assert response.metadata.overall_pass_rate == 1.0
    assert len(response.metadata.dimension_summaries) == 4
    summaries_by_dim = {s.dimension: s for s in response.metadata.dimension_summaries}
    assert summaries_by_dim[QualityDimension.RELEVANCE].evaluated_cases_count == 40
    assert summaries_by_dim[QualityDimension.SAFE_NO_ANSWER].evaluated_cases_count == 20
    assert all(s.threshold_decision_status == ThresholdDecisionStatus.PROPOSED for s in summaries_by_dim.values())
    assert all(s.meets_proposed_threshold is True for s in summaries_by_dim.values())


@pytest.mark.asyncio
async def test_evaluation_api_endpoints_post_and_get():
    eval_service = MagicMock()
    run_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    now = datetime.now(UTC)

    mock_response = {
        "id": str(run_id),
        "prompt_version_id": str(prompt_id),
        "status": "completed",
        "created_at": now.isoformat(),
        "metadata": {
            "dataset": {
                "dataset_id": "mvp-golden-set",
                "dataset_version": "v1.0.0",
                "dataset_path": None,
                "known_limitations": ["Single document corpus"],
            },
            "thresholds": {
                "decision_status": "PROPOSED",
                "relevance_min_score": 0.75,
                "factual_grounding_min_score": 0.85,
                "citation_correctness_min_score": 1.0,
                "target_pass_rate": 0.85,
            },
            "total_cases": 1,
            "passed_cases": 1,
            "failed_cases": 0,
            "overall_pass_rate": 1.0,
            "dimension_summaries": [],
            "followup_jira_issues": [],
            "case_results": [],
        },
    }
    eval_service.run_evaluation = AsyncMock(return_value=mock_response)
    eval_service.get_run = AsyncMock(return_value=mock_response)

    app = FastAPI()
    app.include_router(router)
    install_generation_exception_handlers(app)
    app.state.evaluation_service = eval_service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        post_resp = await client.post(
            "/api/v1/rag/evaluations/runs",
            json={
                "prompt_version_id": str(prompt_id),
                "dataset": {"dataset_id": "mvp-golden-set", "dataset_version": "v1.0.0"},
                "thresholds": {
                    "decision_status": "PROPOSED",
                    "relevance_min_score": 0.75,
                    "factual_grounding_min_score": 0.85,
                    "citation_correctness_min_score": 1.0,
                    "target_pass_rate": 0.85,
                },
            },
        )
        assert post_resp.status_code == 201
        assert post_resp.json()["id"] == str(run_id)

        get_resp = await client.get(f"/api/v1/rag/evaluations/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == str(run_id)
