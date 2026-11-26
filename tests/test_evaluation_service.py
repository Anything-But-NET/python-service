import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from python_service.api.errors import GenerationServiceError
from python_service.api.exception_handlers import install_generation_exception_handlers
from python_service.api.routes import router
from python_service.api.schemas import (
    EvaluationDatasetConfig,
    EvaluationRunCreateRequest,
    EvaluationRunMetadataPayload,
    EvaluationRunStatus,
    ExpectedOutcomeType,
    QualityDimension,
    QualityThresholdsConfig,
    RagCitation,
    RagGenerationResponse,
    RagGenerationStatus,
    ThresholdDecisionStatus,
)
from python_service.config import Settings
from python_service.models.domain import EvaluationRun, PromptVersion
from python_service.services.evaluation_service import (
    EvaluationService,
    _token_support_ratio,
    check_citation_correctness,
    check_factual_grounding,
    check_relevance,
    check_safe_no_answer,
)
from python_service.services.grounded_answer_service import (
    GroundedAnswerService,
    SYSTEM_PROMPT,
)
from python_service.services.parse_service import MarkdownParserService


REAL_CORPUS_MARKDOWN = """# Automated CI/CD Infrastructure Blueprint

This document outlines the technical specification for the automated CI/CD pipeline powering the core backend service infrastructure. The pipeline utilizes Jenkins for execution, Docker for containerization, Amazon ECR for artifact storage, and Amazon EKS (Kubernetes) for runtime orchestration.

## 1. System Architecture Diagram

PRE-FLIGHT / DEVELOPER WORKSTATION: Developer runs git push to GitHub Enterprise Repository.
JENKINS AUTOMATION SERVER runs in AWS EC2 / VPC Private Subnet.
AWS INFRASTRUCTURE ENGINE is located in us-east-1.
AWS ECR Container Registry stores repo.ecr.us-east-1.amazonaws.com/backend/api-service:v1.4.2.
Amazon EKS Cluster runs Kubernetes v1.28 with Ingress Controller (ALB) and ClusterIP Service: api-backend-svc.
Managed Database Layer runs Amazon Aurora PostgreSQL (Private DB Subnet).

## 2. Pipeline Execution Stages

### Stage 1: Checkout & Environment Initialization
- Event-driven trigger execution via GitHub webhook (push event to main or release branches).
- Initialization of ephemeral Jenkins agent pod operating within the AWS EKS build namespace.
- Authentication credentials loaded dynamically via AWS IAM Roles for Service Accounts (IRSA).

### Stage 2: Container Image Construction
- Build executed using Docker CLI (docker buildx) utilizing multi-stage builds to ensure minimal image payload footprint.
- Dockerfile execution optimizes layer caching by separating dependency manifests from source code compilation.

### Stage 3: Static Analysis & Vulnerability Scanning
- Static Code Analysis (SCA) via SonarQube scanner for quality gate enforcement.
- Container security analysis via Trivy scanner executing image CVE evaluation prior to registry push. Pipeline halts if critical CVE threshold is breached.

### Stage 4: Artifact Tagging & Registry Push
- Image tagging follows Semantic Versioning along with the short Git commit SHA: 1.4.2-b8d9e1f.
- Authentication against AWS ECR using temporary token provided via aws ecr get-login-password.
- Image pushed to high-availability target Amazon ECR repository.

### Stage 5: Kubernetes Manifest Interpolation & Deployment
- Helm configuration values dynamically rendered per environment target (Staging vs. Production).
- Zero-downtime Rolling Update deployment strategy executed via helm upgrade --install.
- Automatic readiness probe checking ensures active traffic routing occurs only after container health validation (/healthz endpoint returning HTTP 200).

## 3. Environment Connection Configurations

- Primary Production Read-Write Cluster: postgresql://db_master_admin:a8f9c2d7e1b30495a@aurora-pg-prod-cluster.cluster-c92x81mks.us-east-1.rds.amazonaws.com:5432/core_backend_db?sslmode=verify-full&sslrootcert=/etc/ssl/certs/rds-ca-2021.pem
- Read-Replica Pool (Reporting Services): postgresql://db_read_only:f3b2018a471e9c20d@aurora-pg-prod-ro.cluster-c92x81mks.us-east-1.rds.amazonaws.com:5432/core_backend_db?sslmode=verify-full&target_session_attrs=read-only
- Elasticache Redis Cluster (TLS Enabled): rediss://:eY3xP9vL1qR8zW5mK2nT7sB0aC4vF6gH@prod-redis-cluster.cache.us-east-1.amazonaws.com:6379/0
- Amazon MQ (ActiveMQ) Broker Connection: ssl://b-4910283f-8a02-41e9-918d-3298a0d018a2-1.mq.us-east-1.amazonaws.com:61616?wireFormat.maxFrameSize=104857600
- EKS API Server Endpoint: https://B8E301F29C400A2D8197110C1A39D9B2.gr7.us-east-1.eks.amazonaws.com
- Internal Service Mesh Ingress Endpoint: https://internal-alb-ingress-prod-1029384756.us-east-1.elb.amazonaws.com/api/v1

## 4. Pipeline Infrastructure Specifications

- Source Control: GitHub Enterprise | Branch protection on main, mandatory PR reviews | Multi-region redundant hosting
- Automation Server: Jenkins v2.414 | EC2 Deployment, Docker-in-Docker (dind) execution agents | Dynamic Auto Scaling Group (1-10 instances)
- Container Runtime: Docker v24.0 | Multi-stage alpine-based build containers | Local registry cache enabled
- Artifact Registry: Amazon ECR | Private repository, AES-256 encryption at rest | Multi-AZ replication, lifecycle rules (30 days)
- Orchestrator: Amazon EKS v1.28 | Managed Node Groups (m5.xlarge), Cilium CNI | Auto-scaling groups spanning 3 Availability Zones
- Ingress Control: AWS ALB Ingress | AWS Load Balancer Controller with ACM TLS Termination | Cross-zone load balancing with health checks
"""


def test_token_support_ratio_prevents_numeric_substring_false_positives():
    """Ensure short numeric tokens like '3' or '30' do not match 'stage 3', 'v1.23', or '300'."""
    assert _token_support_ratio("3 Availability Zones", "Stage 2 runs v1.23 with 300 nodes.") == 0.0
    assert _token_support_ratio("30 days", "Timeout is 3000 seconds.") == 0.0
    assert _token_support_ratio("3 Availability Zones", "Auto-scaling groups spanning 3 Availability Zones.") == 1.0


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
            "document_id": "backend_deployment_pipeline_architecture.md",
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


def test_check_citation_correctness_respects_threshold_and_source_matching():
    chunk_id = "123e4567-e89b-12d3-a456-426614174000"
    passages = [
        {
            "document_id": "backend_deployment_pipeline_architecture.md",
            "chunk_id": chunk_id,
            "score": 0.95,
            "text": "Artifacts are stored in Amazon ECR in us-east-1.",
        }
    ]
    good_citations = [
        RagCitation(
            document_id="backend_deployment_pipeline_architecture.md",
            chunk_id=chunk_id,
            score=0.95,
        )
    ]
    res_good = check_citation_correctness(
        actual_answer="Artifacts are stored in Amazon ECR.",
        actual_citations=good_citations,
        retrieved_passages=passages,
        expected_document_ids=["backend_deployment_pipeline_architecture.md"],
        expected_outcome=ExpectedOutcomeType.ANSWER_EXPECTED,
        threshold=1.0,
    )
    assert res_good.passed is True
    assert res_good.score == 1.0

    wrong_doc_citations = [
        RagCitation(
            document_id="completely-unrelated-hr-doc",
            chunk_id=chunk_id,
            score=0.95,
        )
    ]
    # 2 of 3 subchecks pass (citations_exist=True, supports_content=True, resolves_to_expected=False -> score 0.6667)
    res_strict = check_citation_correctness(
        actual_answer="Artifacts are stored in Amazon ECR.",
        actual_citations=wrong_doc_citations,
        retrieved_passages=passages,
        expected_document_ids=["backend_deployment_pipeline_architecture.md"],
        expected_outcome=ExpectedOutcomeType.ANSWER_EXPECTED,
        threshold=1.0,
    )
    assert res_strict.passed is False
    assert res_strict.score == pytest.approx(0.6667, rel=1e-3)

    res_relaxed = check_citation_correctness(
        actual_answer="Artifacts are stored in Amazon ECR.",
        actual_citations=wrong_doc_citations,
        retrieved_passages=passages,
        expected_document_ids=["backend_deployment_pipeline_architecture.md"],
        expected_outcome=ExpectedOutcomeType.ANSWER_EXPECTED,
        threshold=0.6,
    )
    assert res_relaxed.passed is True


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


def test_load_golden_set_blocks_path_traversal_and_validates_metadata(tmp_path: Path):
    db = MagicMock()
    grounded = MagicMock()
    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir()
    valid_file = ds_dir / "golden_set_v1.json"
    valid_file.write_text(
        json.dumps({"dataset_id": "mvp-golden-set", "dataset_version": "v1.0.0", "cases": []}),
        encoding="utf-8",
    )

    service = EvaluationService(db=db, grounded_answer_service=grounded, datasets_dir=ds_dir)

    with pytest.raises(GenerationServiceError) as exc_traversal:
        service.load_golden_set(
            EvaluationDatasetConfig(
                dataset_id="mvp-golden-set",
                dataset_version="v1.0.0",
                dataset_path="../pyproject.toml",
            )
        )
    assert exc_traversal.value.http_status == 400

    with pytest.raises(GenerationServiceError) as exc_mismatch:
        service.load_golden_set(
            EvaluationDatasetConfig(
                dataset_id="wrong-id",
                dataset_version="v9.9.9",
                dataset_path="golden_set_v1.json",
            )
        )
    assert exc_mismatch.value.http_status == 422


@pytest.mark.asyncio
async def test_prompt_override_uses_requested_prompt_version_and_single_retrieval(tmp_path: Path):
    """Verifies Review Issues #1 and #2:
    - requested prompt_version_id's prompt_text and metadata are passed to LLM (bypassing latest cached prompt);
    - vectorizer and db.query_with_auto_merge are called strictly once per case.
    """
    db = MagicMock()
    older_prompt_id = uuid.uuid4()
    older_prompt = PromptVersion(
        id=older_prompt_id,
        prompt_text="CUSTOM_OLDER_PROMPT_TEXT_FOR_EVALUATION",
        metadata={"model": "custom-eval-model-v1", "temperature": 0.1, "max_tokens": 500},
    )
    db.get_prompt_version = AsyncMock(return_value=older_prompt)
    db.get_latest_prompt_version = AsyncMock(
        return_value=PromptVersion(
            id=uuid.uuid4(),
            prompt_text="LATEST_CACHED_PROMPT_THAT_MUST_NOT_BE_USED",
            metadata={"model": "latest-model"},
        )
    )

    chunk_id = "123e4567-e89b-12d3-a456-426614174000"
    db.query_with_auto_merge = AsyncMock(
        return_value=[
            {
                "document_id": "backend_deployment_pipeline_architecture.md",
                "score": 0.91,
                "chunk_ids": [chunk_id],
                "chunks": ["Docker is used for containerization in the CI/CD pipeline."],
            }
        ]
    )

    run_id = uuid.uuid4()
    db.create_evaluation_run = AsyncMock(
        return_value=EvaluationRun(
            id=run_id,
            prompt_version_id=older_prompt_id,
            status="running",
            metadata={},
            created_at=datetime.now(UTC),
        )
    )
    db.update_evaluation_run = AsyncMock(
        side_effect=lambda run_id, status, metadata=None: EvaluationRun(
            id=run_id,
            prompt_version_id=older_prompt_id,
            status=status,
            metadata=metadata or {},
            created_at=datetime.now(UTC),
        )
    )

    vectorizer = MagicMock()
    vectorizer.vectorize_query = MagicMock(return_value=[0.1] * 1024)

    llm = MagicMock()
    llm.generate_json = AsyncMock(
        return_value=json.dumps(
            {
                "answer": "Docker is used for containerization.",
                "confidence_score": 0.93,
                "insufficient_evidence": False,
                "cited_chunk_ids": [chunk_id],
            }
        )
    )

    grounded_service = GroundedAnswerService(
        db=db,
        vectorizer=vectorizer,
        llm=llm,
        settings=Settings(),
    )

    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir()
    (ds_dir / "golden_set_v1.json").write_text(
        json.dumps(
            {
                "dataset_id": "mvp-golden-set",
                "dataset_version": "v1.0.0",
                "cases": [
                    {
                        "case_id": "GS-021",
                        "query": "What tool is used for containerization in the CI/CD pipeline?",
                        "expected_outcome": "ANSWER_EXPECTED",
                        "expected_answer": "Docker",
                        "expected_document_ids": ["backend_deployment_pipeline_architecture.md"],
                        "user_permissions": {"allowed_groups": ["all-employees"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    eval_service = EvaluationService(
        db=db,
        grounded_answer_service=grounded_service,
        datasets_dir=ds_dir,
    )

    req = EvaluationRunCreateRequest(
        prompt_version_id=older_prompt_id,
        dataset=EvaluationDatasetConfig(dataset_id="mvp-golden-set", dataset_version="v1.0.0"),
        thresholds=QualityThresholdsConfig(
            decision_status=ThresholdDecisionStatus.PROPOSED,
            relevance_min_score=0.75,
            factual_grounding_min_score=0.85,
            citation_correctness_min_score=1.0,
            target_pass_rate=0.85,
        ),
    )

    resp = await eval_service.run_evaluation(req)
    assert resp.status == EvaluationRunStatus.COMPLETED
    assert resp.metadata.passed_cases == 1

    # Strictly 1 vectorization and 1 DB retrieval for the 1 case
    assert vectorizer.vectorize_query.call_count == 1
    assert db.query_with_auto_merge.await_count == 1
    db.get_latest_prompt_version.assert_not_awaited()

    # Verify exact prompt_text and metadata kwargs sent to LLM
    llm.generate_json.assert_awaited_once()
    sent_messages = llm.generate_json.call_args.args[0]
    sent_kwargs = llm.generate_json.call_args.kwargs
    assert [message["role"] for message in sent_messages] == ["system", "user"]
    system_content = sent_messages[0]["content"]
    serialized_prompt = json.dumps(
        {"prompt_text": "CUSTOM_OLDER_PROMPT_TEXT_FOR_EVALUATION"},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert system_content.startswith(SYSTEM_PROMPT.rstrip())
    assert serialized_prompt in system_content
    assert system_content.index(serialized_prompt) < system_content.index(
        "FINAL IMMUTABLE SECURITY REMINDER"
    )
    payload_text = sent_messages[1]["content"].split("\n", 1)[1]
    payload = json.loads(payload_text)
    assert "versioned_prompt" not in payload
    assert sent_kwargs == {"model": "custom-eval-model-v1", "temperature": 0.1, "max_tokens": 500}


@pytest.mark.asyncio
async def test_evaluation_service_runs_on_parsed_corpus_and_60_golden_set_cases():
    """Executes the evaluator on all 60 Golden Set cases against chunks parsed by MarkdownParserService
    from the actual SCRUM-6 test document ([Eng] Backend Deployment Pipeline Architecture).
    """
    parser = MarkdownParserService()
    root_node = parser.parse(REAL_CORPUS_MARKDOWN)

    parsed_chunks: list[dict[str, str]] = []

    def _collect_chunks(node):
        if node.content.strip():
            parsed_chunks.append(
                {
                    "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_OID, f"{node.name}:{node.content[:40]}")),
                    "header": node.name,
                    "text": f"## {node.name}\n{node.content.strip()}",
                }
            )
        for child in node.children:
            _collect_chunks(child)

    _collect_chunks(root_node)
    assert len(parsed_chunks) >= 8

    db = MagicMock()
    prompt_id = uuid.uuid4()
    run_id = uuid.uuid4()
    db.get_prompt_version = AsyncMock(
        return_value=PromptVersion(id=prompt_id, prompt_text="system prompt", metadata={"model": "gpt-4o-mini"})
    )
    db.create_evaluation_run = AsyncMock(
        return_value=EvaluationRun(
            id=run_id,
            prompt_version_id=prompt_id,
            status="running",
            metadata={},
            created_at=datetime.now(UTC),
        )
    )
    db.update_evaluation_run = AsyncMock(
        side_effect=lambda run_id, status, metadata=None: EvaluationRun(
            id=run_id,
            prompt_version_id=prompt_id,
            status=status,
            metadata=metadata or {},
            created_at=datetime.now(UTC),
        )
    )

    def _retrieve_top_chunks(query: str) -> list[dict]:
        best = sorted(
            parsed_chunks,
            key=lambda ch: _token_support_ratio(query, ch["text"]),
            reverse=True,
        )[:2]
        top_score = _token_support_ratio(query, best[0]["text"]) if best else 0.0
        if top_score < 0.25:
            return []
        return [
            {
                "document_id": "backend_deployment_pipeline_architecture.md",
                "score": round(min(0.96, 0.65 + 0.3 * top_score), 4),
                "chunk_ids": [ch["chunk_id"] for ch in best],
                "chunks": [ch["text"] for ch in best],
            }
        ]

    vectorizer = MagicMock()
    vectorizer.vectorize_query = MagicMock(side_effect=lambda q: q)

    async def _fake_query_with_auto_merge(vector, allowed_groups, limit, threshold):
        # vector carries the query string from side_effect above
        return _retrieve_top_chunks(str(vector))

    db.query_with_auto_merge = AsyncMock(side_effect=_fake_query_with_auto_merge)

    async def _grounded_llm_from_passages(messages, **kwargs):
        user_content = messages[-1]["content"]
        payload_json = user_content.split("\n", 1)[1]
        data = json.loads(payload_json)
        question = data["question"]
        passages = data["retrieved_passages"]

        best_passage = max(passages, key=lambda p: _token_support_ratio(question, p["text"]))
        if _token_support_ratio(question, best_passage["text"]) < 0.25:
            return json.dumps(
                {
                    "answer": "",
                    "confidence_score": 0.0,
                    "insufficient_evidence": True,
                    "cited_chunk_ids": [],
                }
            )

        lines = [
            ln.strip(" -●*")
            for ln in best_passage["text"].splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        best_line = max(lines, key=lambda ln: _token_support_ratio(question, ln))
        return json.dumps(
            {
                "answer": best_line,
                "confidence_score": 0.91,
                "insufficient_evidence": False,
                "cited_chunk_ids": [best_passage["chunk_id"]],
            }
        )

    llm = MagicMock()
    llm.generate_json = AsyncMock(side_effect=_grounded_llm_from_passages)

    grounded_service = GroundedAnswerService(
        db=db,
        vectorizer=vectorizer,
        llm=llm,
        settings=Settings(),
    )
    eval_service = EvaluationService(db=db, grounded_answer_service=grounded_service)

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
    assert vectorizer.vectorize_query.call_count == 60
    assert db.query_with_auto_merge.await_count == 60

    summaries = {s.dimension: s for s in response.metadata.dimension_summaries}
    assert summaries[QualityDimension.RELEVANCE].evaluated_cases_count == 40
    assert summaries[QualityDimension.SAFE_NO_ANSWER].evaluated_cases_count == 20
    assert response.metadata.overall_pass_rate >= 0.75
    assert response.metadata.failed_cases > 0
    assert any(c.failure_reasons for c in response.metadata.case_results if not c.passed)


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

        # Missing required threshold fields must return 422 (aligned with OpenAPI required fields)
        invalid_resp = await client.post(
            "/api/v1/rag/evaluations/runs",
            json={
                "prompt_version_id": str(prompt_id),
                "dataset": {"dataset_id": "mvp-golden-set", "dataset_version": "v1.0.0"},
                "thresholds": {},
            },
        )
        assert invalid_resp.status_code == 422
        assert invalid_resp.json()["status"] == "VALIDATION_ERROR"

        get_resp = await client.get(f"/api/v1/rag/evaluations/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == str(run_id)


@pytest.mark.asyncio
async def test_running_and_failed_evaluation_runs_persist_valid_metadata_for_get_run(tmp_path: Path):
    """Ensure GET /runs/{run_id} returns schema-valid payload (not 500) even when status is running or failed."""
    stored_runs: dict[uuid.UUID, EvaluationRun] = {}
    prompt_id = uuid.uuid4()
    run_id = uuid.uuid4()

    db = MagicMock()
    db.get_prompt_version = AsyncMock(
        return_value=PromptVersion(id=prompt_id, prompt_text="sys", metadata={"model": "gpt-4o-mini"})
    )

    async def _create_run(prompt_version_id, status, metadata):
        record = EvaluationRun(
            id=run_id,
            prompt_version_id=prompt_version_id,
            status=status,
            metadata=metadata,
            created_at=datetime.now(UTC),
        )
        stored_runs[run_id] = record
        return record

    async def _update_run(run_id, status, metadata=None):
        existing = stored_runs[run_id]
        updated = EvaluationRun(
            id=existing.id,
            prompt_version_id=existing.prompt_version_id,
            status=status,
            metadata=metadata if metadata is not None else existing.metadata,
            created_at=existing.created_at,
        )
        stored_runs[run_id] = updated
        return updated

    async def _get_run(run_id):
        return stored_runs.get(uuid.UUID(str(run_id)))

    db.create_evaluation_run = AsyncMock(side_effect=_create_run)
    db.update_evaluation_run = AsyncMock(side_effect=_update_run)
    db.get_evaluation_run = AsyncMock(side_effect=_get_run)

    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir()
    (ds_dir / "golden_set_v1.json").write_text(
        json.dumps(
            {
                "dataset_id": "mvp-golden-set",
                "dataset_version": "v1.0.0",
                "cases": [
                    {
                        "case_id": "GS-021",
                        "query": "What tool is used for containerization?",
                        "expected_outcome": "ANSWER_EXPECTED",
                        "expected_answer": "Docker",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    grounded = MagicMock()
    eval_service = EvaluationService(db=db, grounded_answer_service=grounded, datasets_dir=ds_dir)
    # Simulate an unexpected runner-level crash after creating the 'running' row
    eval_service._execute_case_against_rag = AsyncMock(side_effect=RuntimeError("unexpected runner crash"))

    req = EvaluationRunCreateRequest(
        prompt_version_id=prompt_id,
        dataset=EvaluationDatasetConfig(dataset_id="mvp-golden-set", dataset_version="v1.0.0"),
        thresholds=QualityThresholdsConfig(
            decision_status=ThresholdDecisionStatus.PROPOSED,
            relevance_min_score=0.75,
            factual_grounding_min_score=0.85,
            citation_correctness_min_score=1.0,
            target_pass_rate=0.85,
        ),
    )

    with pytest.raises(RuntimeError, match="unexpected runner crash"):
        await eval_service.run_evaluation(req)

    # Initial running metadata must already be schema-valid
    initial_meta = db.create_evaluation_run.call_args.kwargs["metadata"]
    EvaluationRunMetadataPayload(**initial_meta)

    # GET on the failed run must return a valid EvaluationRunResponse (HTTP 200), never 500
    app = FastAPI()
    app.include_router(router)
    install_generation_exception_handlers(app)
    app.state.evaluation_service = eval_service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/api/v1/rag/evaluations/runs/{run_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["metadata"]["total_cases"] == 1
    assert body["metadata"]["case_results"] == []

