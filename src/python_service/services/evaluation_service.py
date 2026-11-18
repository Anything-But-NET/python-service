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
from python_service.config import get_settings
from python_service.db.service import RagDbService
from python_service.models.domain import PromptVersion
from python_service.services.grounded_answer_service import GroundedAnswerService
from python_service.services.query_filter_service import DEFAULT_SPAM_RESPONSE

logger = logging.getLogger(__name__)

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

# Maps SCRUM-10 redacted placeholder tokens to concrete tokens present in the source PDF
_PLACEHOLDER_SYNONYMS: dict[str, list[str]] = {
    "db_prod_cluster": ["aurora-pg-prod-cluster", "db_master_admin", "core_backend_db"],
    "redis_prod_cluster": ["prod-redis-cluster", "rediss", "6379"],
    "eks_endpoint": ["b8e301f29c400a2d8197110c1a39d9b2", "eks.amazonaws.com"],
    "alb_endpoint": ["internal-alb-ingress-prod-1029384756", "elb.amazonaws.com"],
}


def _resolve_datasets_dir(configured_dir: str | Path | None = None) -> Path:
    raw = configured_dir if configured_dir is not None else get_settings().datasets_dir
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (Path(__file__).resolve().parent.parent.parent.parent / candidate).resolve()
    return repo_candidate


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_./=-]+", text.lower())
    cleaned = [t.strip("._-/=") for t in tokens if t.strip("._-/=")]
    content = [t for t in cleaned if t not in _STOPWORDS]
    return content if content else cleaned


def _expand_token_set(tokens: list[str]) -> set[str]:
    expanded: set[str] = set(tokens)
    for tok in tokens:
        for part in re.split(r"[./=:-]+", tok):
            part_clean = part.strip("._")
            if part_clean and part_clean not in _STOPWORDS:
                expanded.add(part_clean)
    return expanded


def _token_matches(target_tok: str, source_token_set: set[str]) -> bool:
    if target_tok in source_token_set:
        return True
    synonyms = _PLACEHOLDER_SYNONYMS.get(target_tok)
    if synonyms and any(syn in source_token_set for syn in synonyms):
        return True
    # Allow morphological prefix match only for alphabetic words >= 6 chars (e.g. "separating" vs "separate")
    if len(target_tok) >= 6 and target_tok.isalpha():
        stem = target_tok[:6]
        return any(
            len(src_tok) >= 6 and src_tok.isalpha() and src_tok.startswith(stem)
            for src_tok in source_token_set
        )
    return False


def _token_support_ratio(target_text: str, source_text: str) -> float:
    """Calculate token recall using exact token set matching (no raw substring false positives)."""
    target_tokens = _tokenize(target_text)
    if not target_tokens:
        return 0.0
    source_token_set = _expand_token_set(_tokenize(source_text))
    if not source_token_set:
        return 0.0
    matched = sum(1 for tok in target_tokens if _token_matches(tok, source_token_set))
    return round(matched / len(target_tokens), 4)


def _split_claims(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part and part.strip()]


def _doc_id_matches_expected(cited_doc_id: str, expected_doc_ids: list[str]) -> bool:
    cited_norm = cited_doc_id.strip().lower()
    if not cited_norm:
        return False
    cited_slug = re.sub(r"[^a-z0-9]+", "-", cited_norm).strip("-")
    for exp in expected_doc_ids:
        exp_norm = exp.strip().lower()
        exp_slug = re.sub(r"[^a-z0-9]+", "-", exp_norm).strip("-")
        if (
            cited_norm == exp_norm
            or cited_slug == exp_slug
            or (len(exp_slug) >= 12 and (exp_slug in cited_slug or cited_slug in exp_slug))
        ):
            return True
    return False


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
    score = round(
        max(exp_score, 0.8 * exp_score + 0.2 * query_score) if expected_answer else query_score,
        4,
    )
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
    combined_passages = "\n".join(
        str(p.get("text", "")) for p in retrieved_passages if p.get("text")
    )
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
        return QualityCheckResult(
            applicable=False,
            passed=True,
            score=1.0,
            threshold=threshold,
            calculation_method=method,
            details={"reason": "Not applicable for INSUFFICIENT_EVIDENCE cases"},
        )

    citations_exist = len(actual_citations) > 0
    cited_doc_ids = sorted({c.document_id for c in actual_citations})

    if expected_document_ids:
        unexpected_docs = [
            doc_id
            for doc_id in cited_doc_ids
            if not _doc_id_matches_expected(doc_id, expected_document_ids)
        ]
        resolves_to_expected = bool(cited_doc_ids) and len(unexpected_docs) == 0
        missing_docs = [] if resolves_to_expected else list(expected_document_ids[:1])
    else:
        resolves_to_expected = citations_exist
        unexpected_docs = []
        missing_docs = []

    passage_by_chunk = {
        str(p.get("chunk_id")): str(p.get("text", "")) for p in retrieved_passages
    }
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
    passed = citations_exist and score >= threshold

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
        datasets_dir: Path | str | None = None,
        max_concurrency: int = 5,
    ) -> None:
        self.db = db
        self.grounded_answer_service = grounded_answer_service
        self.datasets_dir = _resolve_datasets_dir(datasets_dir)
        self.max_concurrency = max(1, max_concurrency)

    def load_golden_set(self, dataset_config: EvaluationDatasetConfig) -> dict[str, Any]:
        allowed_dir = self.datasets_dir.resolve()
        if dataset_config.dataset_path:
            raw_path = Path(dataset_config.dataset_path)
            # If the caller passed "datasets/golden_set_v1.json", strip the leading "datasets/" prefix
            if not raw_path.is_absolute() and raw_path.parts and raw_path.parts[0] == allowed_dir.name:
                raw_path = Path(*raw_path.parts[1:]) if len(raw_path.parts) > 1 else Path("golden_set_v1.json")
            candidate = (
                raw_path.resolve()
                if raw_path.is_absolute()
                else (allowed_dir / raw_path).resolve()
            )
        else:
            candidate = (allowed_dir / "golden_set_v1.json").resolve()

        if not candidate.is_relative_to(allowed_dir):
            raise GenerationServiceError(
                GenerationErrorStatus.VALIDATION_ERROR,
                "Invalid dataset_path: path traversal outside the datasets directory is forbidden",
                http_status=400,
            )

        if not candidate.exists() or not candidate.is_file():
            raise GenerationServiceError(
                GenerationErrorStatus.VALIDATION_ERROR,
                f"Golden Set dataset file not found: {candidate.name}",
                http_status=404,
            )

        with open(candidate, "r", encoding="utf-8") as f:
            raw_dataset = json.load(f)

        file_dataset_id = raw_dataset.get("dataset_id")
        file_dataset_version = raw_dataset.get("dataset_version")
        if (
            file_dataset_id != dataset_config.dataset_id
            or file_dataset_version != dataset_config.dataset_version
        ):
            raise GenerationServiceError(
                GenerationErrorStatus.VALIDATION_ERROR,
                (
                    f"Dataset metadata mismatch: requested "
                    f"{dataset_config.dataset_id}@{dataset_config.dataset_version}, "
                    f"but file contains {file_dataset_id}@{file_dataset_version}"
                ),
                http_status=422,
            )

        return raw_dataset

    def evaluate_single_case(
        self,
        case: dict[str, Any],
        actual_status: RagGenerationStatus,
        actual_answer: str | None,
        actual_confidence: float,
        actual_citations: list[RagCitation],
        retrieved_passages: list[dict[str, Any]],
        thresholds: QualityThresholdsConfig,
        execution_error: str | None = None,
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
        if execution_error:
            failure_reasons.append(f"execution_error: {execution_error}")

        checks_map = {
            "relevance": rel_check,
            "factual_grounding": fg_check,
            "citation_correctness": cit_check,
            "safe_no_answer": sna_check,
        }

        if expected_outcome == ExpectedOutcomeType.ANSWER_EXPECTED and sna_check.details:
            if sna_check.details.get("unnecessary_abstention"):
                failure_reasons.append(
                    f"safe_no_answer: unnecessary abstention (status={actual_status.value})"
                )

        applicable_checks = [chk for chk in checks_map.values() if chk.applicable]
        for name, chk in checks_map.items():
            if chk.applicable and not chk.passed:
                failure_reasons.append(
                    f"{name}: score {chk.score:.2f} < threshold {chk.threshold:.2f}"
                )

        passed = (
            execution_error is None
            and actual_status != RagGenerationStatus.ERROR
            and all(chk.passed for chk in applicable_checks)
        )

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
        prompt_version: PromptVersion,
    ) -> tuple[
        RagGenerationStatus,
        str | None,
        float,
        list[RagCitation],
        list[dict[str, Any]],
        str | None,
    ]:
        """Run a Golden Set case through single-pass retrieval + grounded generation."""
        query = str(case["query"])
        perms_raw = case.get("user_permissions") or {"allowed_groups": ["all-employees"]}
        permissions = UserPermissions(**perms_raw)

        try:
            gen_resp, passages = await self.grounded_answer_service.generate_with_evidence(
                RagGenerationRequest(
                    query_id=f"eval-{case['case_id']}",
                    query=query,
                    user_permissions=permissions,
                ),
                prompt_override=prompt_version,
            )
            return (
                gen_resp.status,
                gen_resp.answer,
                gen_resp.confidence_score,
                gen_resp.citations,
                passages,
                None,
            )
        except Exception as exc:
            logger.warning("Evaluation case %s failed with error: %s", case.get("case_id"), exc)
            return (
                RagGenerationStatus.ERROR,
                None,
                0.0,
                [],
                [],
                str(exc),
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

        running_record = await self.db.create_evaluation_run(
            prompt_version_id=request.prompt_version_id,
            status=EvaluationRunStatus.RUNNING.value,
            metadata={},
        )

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _eval_with_limit(case: dict[str, Any]) -> CaseEvaluationResult:
            async with semaphore:
                (
                    status,
                    answer,
                    confidence,
                    citations,
                    passages,
                    exec_err,
                ) = await self._execute_case_against_rag(case, prompt)
                return self.evaluate_single_case(
                    case=case,
                    actual_status=status,
                    actual_answer=answer,
                    actual_confidence=confidence,
                    actual_citations=citations,
                    retrieved_passages=passages,
                    thresholds=request.thresholds,
                    execution_error=exec_err,
                )

        try:
            case_results = list(
                await asyncio.gather(*(_eval_with_limit(case) for case in raw_cases))
            )

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
                followup_jira_issues=list(raw_dataset.get("followup_jira_issues") or []),
                case_results=case_results,
            )

            updated_run = await self.db.update_evaluation_run(
                run_id=running_record.id,
                status=EvaluationRunStatus.COMPLETED.value,
                metadata=metadata_payload.model_dump(mode="json"),
            )
            final_run = updated_run or running_record

            return EvaluationRunResponse(
                id=final_run.id,
                prompt_version_id=final_run.prompt_version_id,
                status=EvaluationRunStatus.COMPLETED,
                metadata=metadata_payload,
                created_at=final_run.created_at,
            )
        except Exception:
            await self.db.update_evaluation_run(
                run_id=running_record.id,
                status=EvaluationRunStatus.FAILED.value,
            )
            raise

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
