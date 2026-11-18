from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from python_service.api.errors import GenerationServiceError
from python_service.api.schemas import (
    CaseEvaluationResult,
    CaseQualityChecks,
    DimensionAggregateSummary,
    EvaluationDatasetConfig,
    EvaluationRunCreateRequest,
    EvaluationRunMetadataPayload,
    EvaluationRunResponse,
    EvaluationRunStatus,
    ExpectedOutcomeType,
    GenerationErrorStatus,
    QualityCheckResult,
    QualityDimension,
    QualityThresholdsConfig,
    RagCitation,
    RagGenerationRequest,
    RagGenerationStatus,
    UserPermissions,
)
from python_service.db.service import RagDbService
from python_service.services.grounded_answer_service import GroundedAnswerService
from python_service.services.query_filter_service import DEFAULT_SPAM_RESPONSE

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "golden_set_v1.json"
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "how", "i", "if", "in", "into", "is", "it", "my", "of", "on", "or",
    "our", "the", "this", "to", "what", "when", "where", "which", "who",
    "whom", "why", "will", "with", "can", "you", "do", "get", "gets", "used",
}

_ABSTENTION_PHRASES = (
    DEFAULT_SPAM_RESPONSE.lower(),
    "couldn't find information",
    "could not find information",
    "insufficient evidence",
    "i don't have enough information",
    "no relevant information",
)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_./-]+", text.lower())
    cleaned = [t.strip("._-/") for t in tokens if t.strip("._-/")]
    content = [t for t in cleaned if t not in _STOPWORDS]
    return content if content else cleaned


def _split_claims(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part and part.strip()]


def _token_support_ratio(target_text: str, source_text: str) -> float:
    target_tokens = _tokenize(target_text)
    if not target_tokens:
        return 0.0
    source_lower = source_text.lower()
    source_tokens = set(_tokenize(source_text))
    matched = sum(
        1 for tok in target_tokens if tok in source_tokens or tok in source_lower
    )
    return round(matched / len(target_tokens), 4)


def check_relevance(
    query: str,
    expected_answer: str | None,
    actual_answer: str | None,
    expected_outcome: ExpectedOutcomeType,
    threshold: float,
) -> QualityCheckResult:
    """Evaluate answer relevance against expected answer and user question."""
    method = "expected_answer_and_query_token_recall_v1"
    if expected_outcome == ExpectedOutcomeType.INSUFFICIENT_EVIDENCE:
        return QualityCheckResult(
            applicable=False,
            passed=True,
            score=1.0,
            threshold=threshold,
            calculation_method=method,
            details={"reason": "Not applicable for INSUFFICIENT_EVIDENCE cases"},
        )

    answer = (actual_answer or "").strip()
    if not answer or any(phrase in answer.lower() for phrase in _ABSTENTION_PHRASES):
        return QualityCheckResult(
            applicable=True,
            passed=False,
            score=0.0,
            threshold=threshold,
            calculation_method=method,
            details={"reason": "Empty answer or abstention when an answer was expected"},
        )

    exp_score = _token_support_ratio(expected_answer, answer) if expected_answer else 0.0
    query_score = _token_support_ratio(query, answer)
    score = round(max(exp_score, 0.8 * exp_score + 0.2 * query_score) if expected_answer else query_score, 4)
    score = min(1.0, max(0.0, score))

    return QualityCheckResult(
        applicable=True,
        passed=score >= threshold,
        score=score,
        threshold=threshold,
        calculation_method=method,
        details={
            "expected_answer_recall": exp_score,
            "query_token_recall": query_score,
        },
    )


def check_factual_grounding(
    actual_answer: str | None,
    retrieved_passages: list[dict[str, Any]],
    expected_outcome: ExpectedOutcomeType,
    threshold: float,
) -> QualityCheckResult:
    """Evaluate whether claims in the generated answer are supported by retrieved passages."""
    method = "retrieved_passage_claim_support_ratio_v1"
    if expected_outcome == ExpectedOutcomeType.INSUFFICIENT_EVIDENCE:
        return QualityCheckResult(
            applicable=False,
            passed=True,
            score=1.0,
            threshold=threshold,
            calculation_method=method,
            details={"reason": "Not applicable for INSUFFICIENT_EVIDENCE cases"},
        )

    answer = (actual_answer or "").strip()
    combined_passages = "\n".join(str(p.get("text", "")) for p in retrieved_passages if p.get("text"))
    claims = _split_claims(answer)

    if not claims or not combined_passages.strip():
        return QualityCheckResult(
            applicable=True,
            passed=False,
            score=0.0,
            threshold=threshold,
            calculation_method=method,
            details={
                "supported_claims_count": 0,
                "total_claims_count": len(claims),
                "unsupported_claims": claims,
            },
        )

    supported = 0
    unsupported_claims: list[str] = []
    for claim in claims:
        claim_ratio = _token_support_ratio(claim, combined_passages)
        if claim_ratio >= 0.6:
            supported += 1
        else:
            unsupported_claims.append(claim)

    score = round(supported / len(claims), 4)
    return QualityCheckResult(
        applicable=True,
        passed=score >= threshold,
        score=score,
        threshold=threshold,
        calculation_method=method,
        details={
            "supported_claims_count": supported,
            "total_claims_count": len(claims),
            "unsupported_claims": unsupported_claims,
        },
    )


def check_citation_correctness(
    actual_answer: str | None,
    actual_citations: list[RagCitation],
    retrieved_passages: list[dict[str, Any]],
    expected_document_ids: list[str],
    expected_outcome: ExpectedOutcomeType,
    threshold: float,
) -> QualityCheckResult:
    """Verify that citations exist, resolve to expected sources, and support the answer."""
    method = "citation_existence_source_and_support_v1"
    if expected_outcome == ExpectedOutcomeType.INSUFFICIENT_EVIDENCE:
        no_hallucinated_citations = len(actual_citations) == 0
        return QualityCheckResult(
            applicable=False,
            passed=no_hallucinated_citations,
            score=1.0 if no_hallucinated_citations else 0.0,
            threshold=threshold,
            calculation_method=method,
            details={"citations_exist": len(actual_citations) > 0},
        )

    citations_exist = len(actual_citations) > 0
    cited_doc_ids = {c.document_id for c in actual_citations}
    expected_set = set(expected_document_ids)

    if expected_set:
        resolves_to_expected = bool(cited_doc_ids) and cited_doc_ids.issubset(expected_set)
        unexpected_docs = sorted(cited_doc_ids - expected_set)
        missing_docs = sorted(expected_set - cited_doc_ids)
    else:
        resolves_to_expected = citations_exist
        unexpected_docs = []
        missing_docs = []

    passage_by_chunk = {str(p.get("chunk_id")): str(p.get("text", "")) for p in retrieved_passages}
    cited_texts = [
        passage_by_chunk[c.chunk_id]
        for c in actual_citations
        if c.chunk_id in passage_by_chunk
    ]
    combined_cited_text = "\n".join(cited_texts)
    answer = (actual_answer or "").strip()

    support_ratio = (
        _token_support_ratio(answer, combined_cited_text)
        if answer and combined_cited_text
        else 0.0
    )
    all_chunks_in_retrieved = citations_exist and len(cited_texts) == len(actual_citations)
    supports_content = all_chunks_in_retrieved and support_ratio >= 0.5

    sub_checks = [citations_exist, resolves_to_expected, supports_content]
    score = round(sum(1.0 for ok in sub_checks if ok) / 3.0, 4)
    passed = all(sub_checks) and score >= threshold

    return QualityCheckResult(
        applicable=True,
        passed=passed,
        score=score,
        threshold=threshold,
        calculation_method=method,
        details={
            "citations_exist": citations_exist,
            "resolves_to_expected_source": resolves_to_expected,
            "supports_answer_content": supports_content,
            "unexpected_document_ids": unexpected_docs,
            "missing_expected_document_ids": missing_docs,
        },
    )


def check_safe_no_answer(
    expected_outcome: ExpectedOutcomeType,
    actual_status: RagGenerationStatus,
    actual_answer: str | None,
    actual_citations: list[RagCitation],
) -> QualityCheckResult:
    """Verify safe abstention on unanswerable questions and detect unnecessary abstentions."""
    method = "abstention_status_and_hallucination_guard_v1"
    answer_text = (actual_answer or "").strip()
    is_abstention_text = not answer_text or any(
        phrase in answer_text.lower() for phrase in _ABSTENTION_PHRASES
    )
    actual_no_answer = (
        actual_status in (RagGenerationStatus.EMPTY_RETRIEVAL, RagGenerationStatus.LOW_CONFIDENCE)
        or is_abstention_text
    ) and len(actual_citations) == 0

    expected_no_answer = expected_outcome == ExpectedOutcomeType.INSUFFICIENT_EVIDENCE
    if expected_no_answer:
        passed = actual_no_answer and actual_status != RagGenerationStatus.ERROR
        return QualityCheckResult(
            applicable=True,
            passed=passed,
            score=1.0 if passed else 0.0,
            threshold=1.0,
            calculation_method=method,
            details={
                "expected_no_answer": True,
                "actual_no_answer": actual_no_answer,
                "actual_generation_status": actual_status.value,
            },
        )

    # For ANSWER_EXPECTED cases, record whether an unnecessary abstention occurred
    unnecessary_abstention = actual_no_answer
    return QualityCheckResult(
        applicable=False,
        passed=not unnecessary_abstention,
        score=0.0 if unnecessary_abstention else 1.0,
        threshold=1.0,
        calculation_method=method,
        details={
            "expected_no_answer": False,
            "actual_no_answer": actual_no_answer,
            "unnecessary_abstention": unnecessary_abstention,
            "actual_generation_status": actual_status.value,
        },
    )


class EvaluationService:
    """Executes repeatable MVP answer-quality checks on the Golden Set (SCRUM-66 / SCRUM-163)."""

    def __init__(
        self,
        db: RagDbService,
        grounded_answer_service: GroundedAnswerService,
        default_dataset_path: Path = DEFAULT_DATASET_PATH,
    ) -> None:
        self.db = db
        self.grounded_answer_service = grounded_answer_service
        self.default_dataset_path = default_dataset_path

    def load_golden_set(self, dataset_config: EvaluationDatasetConfig) -> dict[str, Any]:
        candidate = (
            Path(dataset_config.dataset_path)
            if dataset_config.dataset_path
            else self.default_dataset_path
        )
        if not candidate.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            candidate = (repo_root / candidate).resolve()

        if not candidate.exists():
            raise GenerationServiceError(
                GenerationErrorStatus.VALIDATION_ERROR,
                f"Golden Set dataset file not found: {dataset_config.dataset_path or candidate.name}",
                http_status=404,
            )

        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_single_case(
        self,
        case: dict[str, Any],
        actual_status: RagGenerationStatus,
        actual_answer: str | None,
        actual_confidence: float,
        actual_citations: list[RagCitation],
        retrieved_passages: list[dict[str, Any]],
        thresholds: QualityThresholdsConfig,
    ) -> CaseEvaluationResult:
        case_id = str(case["case_id"])
        query = str(case["query"])
        expected_outcome = ExpectedOutcomeType(case["expected_outcome"])
        expected_answer = case.get("expected_answer")
        expected_docs = list(case.get("expected_document_ids") or [])

        rel_check = check_relevance(
            query=query,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            expected_outcome=expected_outcome,
            threshold=thresholds.relevance_min_score,
        )
        fg_check = check_factual_grounding(
            actual_answer=actual_answer,
            retrieved_passages=retrieved_passages,
            expected_outcome=expected_outcome,
            threshold=thresholds.factual_grounding_min_score,
        )
        cit_check = check_citation_correctness(
            actual_answer=actual_answer,
            actual_citations=actual_citations,
            retrieved_passages=retrieved_passages,
            expected_document_ids=expected_docs,
            expected_outcome=expected_outcome,
            threshold=thresholds.citation_correctness_min_score,
        )
        sna_check = check_safe_no_answer(
            expected_outcome=expected_outcome,
            actual_status=actual_status,
            actual_answer=actual_answer,
            actual_citations=actual_citations,
        )

        failure_reasons: list[str] = []
        if expected_outcome == ExpectedOutcomeType.ANSWER_EXPECTED:
            if sna_check.details and sna_check.details.get("unnecessary_abstention"):
                failure_reasons.append(
                    f"safe_no_answer: unnecessary abstention (status={actual_status.value})"
                )
            if not rel_check.passed:
                failure_reasons.append(
                    f"relevance: score {rel_check.score:.2f} < threshold {rel_check.threshold:.2f}"
                )
            if not fg_check.passed:
                failure_reasons.append(
                    f"factual_grounding: score {fg_check.score:.2f} < threshold {fg_check.threshold:.2f}"
                )
            if not cit_check.passed:
                failure_reasons.append(
                    f"citation_correctness: score {cit_check.score:.2f} < threshold {cit_check.threshold:.2f}"
                )
            passed = rel_check.passed and fg_check.passed and cit_check.passed
        else:
            if not sna_check.passed:
                failure_reasons.append(
                    f"safe_no_answer: expected abstention for unanswerable query, got status={actual_status.value}"
                )
            if not cit_check.passed:
                failure_reasons.append(
                    "citation_correctness: hallucinated citations returned for unanswerable query"
                )
            passed = sna_check.passed and cit_check.passed

        return CaseEvaluationResult(
            case_id=case_id,
            query=query,
            expected_outcome=expected_outcome,
            expected_document_ids=expected_docs,
            actual_status=actual_status,
            actual_answer=actual_answer,
            actual_confidence_score=round(float(actual_confidence), 4),
            actual_citations=actual_citations,
            passed=passed,
            checks=CaseQualityChecks(
                relevance=rel_check,
                factual_grounding=fg_check,
                citation_correctness=cit_check,
                safe_no_answer=sna_check,
            ),
            failure_reasons=failure_reasons,
        )

    @staticmethod
    def build_dimension_summaries(
        case_results: list[CaseEvaluationResult],
        thresholds: QualityThresholdsConfig,
    ) -> list[DimensionAggregateSummary]:
        dimension_thresholds = {
            QualityDimension.RELEVANCE: thresholds.relevance_min_score,
            QualityDimension.FACTUAL_GROUNDING: thresholds.factual_grounding_min_score,
            QualityDimension.CITATION_CORRECTNESS: thresholds.citation_correctness_min_score,
            QualityDimension.SAFE_NO_ANSWER: 1.0,
        }

        summaries: list[DimensionAggregateSummary] = []
        for dim, prop_score in dimension_thresholds.items():
            applicable_checks = [
                getattr(cr.checks, dim.value)
                for cr in case_results
                if getattr(cr.checks, dim.value).applicable
            ]
            evaluated_count = len(applicable_checks)
            passed_count = sum(1 for c in applicable_checks if c.passed)
            pass_rate = round(passed_count / evaluated_count, 4) if evaluated_count else 0.0
            mean_score = (
                round(sum(c.score for c in applicable_checks) / evaluated_count, 4)
                if evaluated_count
                else 0.0
            )
            meets_threshold = (
                evaluated_count > 0 and pass_rate >= thresholds.target_pass_rate
            )

            summaries.append(
                DimensionAggregateSummary(
                    dimension=dim,
                    evaluated_cases_count=evaluated_count,
                    passed_cases_count=passed_count,
                    observed_pass_rate=pass_rate,
                    observed_mean_score=mean_score,
                    proposed_threshold_score=prop_score,
                    proposed_target_pass_rate=thresholds.target_pass_rate,
                    threshold_decision_status=thresholds.decision_status,
                    meets_proposed_threshold=meets_threshold,
                )
            )
        return summaries

    async def _execute_case_against_rag(
        self,
        case: dict[str, Any],
    ) -> tuple[RagGenerationStatus, str | None, float, list[RagCitation], list[dict[str, Any]]]:
        """Run a Golden Set case through retrieval + grounded answer generation."""
        if "simulated_response" in case:
            sim = case["simulated_response"]
            citations = [RagCitation(**c) for c in sim.get("citations", [])]
            return (
                RagGenerationStatus(sim["status"]),
                sim.get("answer"),
                float(sim.get("confidence_score", 0.0)),
                citations,
                list(sim.get("retrieved_passages", [])),
            )

        query = str(case["query"])
        perms_raw = case.get("user_permissions") or {"allowed_groups": ["All Staff"]}
        permissions = UserPermissions(**perms_raw)

        query_vec = await asyncio.to_thread(
            self.grounded_answer_service.vectorizer.vectorize_query,
            query,
        )
        raw_results = await self.db.query_with_auto_merge(
            vector=query_vec,
            allowed_groups=permissions.allowed_groups,
            limit=self.grounded_answer_service.settings.rag_retrieval_limit,
            threshold=self.grounded_answer_service.settings.rag_auto_merge_threshold,
        )
        passages = self.grounded_answer_service._prepare_passages(raw_results)

        gen_resp = await self.grounded_answer_service.generate(
            RagGenerationRequest(
                query_id=f"eval-{case['case_id']}",
                query=query,
                user_permissions=permissions,
            )
        )
        return (
            gen_resp.status,
            gen_resp.answer,
            gen_resp.confidence_score,
            gen_resp.citations,
            passages,
        )

    async def run_evaluation(
        self,
        request: EvaluationRunCreateRequest,
    ) -> EvaluationRunResponse:
        prompt = await self.db.get_prompt_version(request.prompt_version_id)
        if prompt is None:
            raise GenerationServiceError(
                GenerationErrorStatus.VALIDATION_ERROR,
                f"Prompt version {request.prompt_version_id} not found",
                http_status=404,
            )

        raw_dataset = self.load_golden_set(request.dataset)
        raw_cases: list[dict[str, Any]] = raw_dataset.get("cases", [])

        known_limitations = list(
            request.dataset.known_limitations
            or raw_dataset.get("known_limitations")
            or []
        )
        dataset_cfg = EvaluationDatasetConfig(
            dataset_id=request.dataset.dataset_id,
            dataset_version=request.dataset.dataset_version,
            dataset_path=request.dataset.dataset_path,
            known_limitations=known_limitations,
        )

        case_results: list[CaseEvaluationResult] = []
        for case in raw_cases:
            status, answer, confidence, citations, passages = await self._execute_case_against_rag(
                case
            )
            result = self.evaluate_single_case(
                case=case,
                actual_status=status,
                actual_answer=answer,
                actual_confidence=confidence,
                actual_citations=citations,
                retrieved_passages=passages,
                thresholds=request.thresholds,
            )
            case_results.append(result)

        total_cases = len(case_results)
        passed_cases = sum(1 for c in case_results if c.passed)
        failed_cases = total_cases - passed_cases
        overall_pass_rate = round(passed_cases / total_cases, 4) if total_cases else 0.0

        dimension_summaries = self.build_dimension_summaries(
            case_results=case_results,
            thresholds=request.thresholds,
        )

        metadata_payload = EvaluationRunMetadataPayload(
            dataset=dataset_cfg,
            thresholds=request.thresholds,
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            overall_pass_rate=overall_pass_rate,
            dimension_summaries=dimension_summaries,
            followup_jira_issues=[],
            case_results=case_results,
        )

        stored_run = await self.db.create_evaluation_run(
            prompt_version_id=request.prompt_version_id,
            status=EvaluationRunStatus.COMPLETED.value,
            metadata=metadata_payload.model_dump(mode="json"),
        )

        return EvaluationRunResponse(
            id=stored_run.id,
            prompt_version_id=stored_run.prompt_version_id,
            status=EvaluationRunStatus(stored_run.status),
            metadata=metadata_payload,
            created_at=stored_run.created_at,
        )

    async def get_run(self, run_id: str) -> EvaluationRunResponse:
        stored = await self.db.get_evaluation_run(run_id)
        if stored is None:
            raise GenerationServiceError(
                GenerationErrorStatus.VALIDATION_ERROR,
                f"Evaluation run {run_id} not found",
                http_status=404,
            )
        return EvaluationRunResponse(
            id=stored.id,
            prompt_version_id=stored.prompt_version_id,
            status=EvaluationRunStatus(stored.status),
            metadata=EvaluationRunMetadataPayload(**stored.metadata),
            created_at=stored.created_at,
        )
