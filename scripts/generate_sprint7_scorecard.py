from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

ANSWER = "ANSWER_WITH_CITATION"
SAFE_NO_ANSWER = "SAFE_NO_ANSWER"
ESCALATE = "ESCALATE_TO_HUMAN"
ABSTENTIONS = {"ABSTAIN_EMPTY_RETRIEVAL", "ABSTAIN_LOW_CONFIDENCE"}


def classify_actual_decision(result: dict[str, Any]) -> str:
    status = str(result.get("actual_status", "ERROR"))
    answer = str(result.get("actual_answer") or "").strip()
    citations = result.get("actual_citations") or []
    if status == "ERROR":
        return "ERROR"
    if status == "ANSWER_GENERATED":
        return "ANSWER_WITH_CITATION" if answer and citations else "ANSWER_WITHOUT_CITATION"
    if status == "EMPTY_RETRIEVAL":
        return "ABSTAIN_EMPTY_RETRIEVAL"
    if status == "LOW_CONFIDENCE":
        return "ABSTAIN_LOW_CONFIDENCE"
    return "UNKNOWN"


def summarize_decisions(dataset: dict[str, Any], evaluation_cases: list[dict[str, Any]]) -> dict[str, Any]:
    results_by_id = {str(item.get("case_id")): item for item in evaluation_cases}
    expected_counts = {ANSWER: 0, SAFE_NO_ANSWER: 0, ESCALATE: 0}
    actual_counts: dict[str, int] = {}
    correct = 0
    false_answers = 0
    unnecessary_abstentions = 0
    escalation_abstentions = 0
    matched = 0

    for case in dataset.get("cases", []):
        case_id = str(case["case_id"])
        result = results_by_id.get(case_id)
        if result is None:
            continue
        matched += 1
        expected = str(case.get("expected_action", ""))
        if expected not in expected_counts:
            raise ValueError(f"Unknown expected_action for {case_id}: {expected!r}")
        expected_counts[expected] += 1
        actual = classify_actual_decision(result)
        actual_counts[actual] = actual_counts.get(actual, 0) + 1

        if expected == ANSWER:
            case_correct = actual == "ANSWER_WITH_CITATION"
            if actual in ABSTENTIONS or actual in {"ERROR", "UNKNOWN"}:
                unnecessary_abstentions += 1
        else:
            case_correct = actual in ABSTENTIONS
            if actual in {"ANSWER_WITH_CITATION", "ANSWER_WITHOUT_CITATION"}:
                false_answers += 1
            if expected == ESCALATE and actual in ABSTENTIONS:
                escalation_abstentions += 1

        if case_correct:
            correct += 1

    return {
        "matched_cases": matched,
        "expected_action_counts": expected_counts,
        "actual_decision_counts": actual_counts,
        "correct_decisions": correct,
        "decision_pass_rate": round(correct / matched, 4) if matched else 0.0,
        "false_answers": false_answers,
        "unnecessary_abstentions": unnecessary_abstentions,
        "expected_escalations": expected_counts[ESCALATE],
        "escalation_abstentions": escalation_abstentions,
        "escalation_delivery_verification": "NOT_VERIFIABLE_IN_PYTHON_SERVICE",
    }


def analyze_confidence(dataset: dict[str, Any], evaluation_cases: list[dict[str, Any]], current_threshold: float) -> dict[str, Any]:
    results_by_id = {str(item.get("case_id")): item for item in evaluation_cases}
    answerable: list[float] = []
    insufficient: list[float] = []

    for case in dataset.get("cases", []):
        result = results_by_id.get(str(case["case_id"]))
        if result is None:
            continue
        score = float(result.get("actual_confidence_score", 0.0))
        score = min(1.0, max(0.0, score))
        if case.get("expected_action") == ANSWER:
            answerable.append(score)
        else:
            insufficient.append(score)

    def stats(values: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "min": round(min(values), 4) if values else None,
            "max": round(max(values), 4) if values else None,
            "mean": round(statistics.mean(values), 4) if values else None,
        }

    answer_stats = stats(answerable)
    insufficient_stats = stats(insufficient)
    enough_data = bool(answerable and insufficient)
    overlap = None
    candidate = None
    if enough_data:
        overlap = max(insufficient) >= min(answerable)
        if not overlap:
            candidate = round((max(insufficient) + min(answerable)) / 2, 4)

    if not enough_data:
        recommendation = "INSUFFICIENT_DATA"
    elif overlap:
        recommendation = "NO_SEPARATING_THRESHOLD"
    else:
        recommendation = "CANDIDATE_THRESHOLD_REQUIRES_REVIEW"

    return {
        "current_runtime_threshold": round(current_threshold, 4),
        "answerable_confidence": answer_stats,
        "insufficient_evidence_confidence": insufficient_stats,
        "confidence_ranges_overlap": overlap,
        "proposed_threshold": candidate,
        "recommendation": recommendation,
        "decision_status": "PROPOSED",
        "method": "observed_confidence_range_separation_v1",
        "limitation": "This is a distribution diagnostic, not a replay of generation under alternative thresholds.",
    }


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def build_scorecard(
    run_payload: dict[str, Any],
    dataset: dict[str, Any],
    operations: dict[str, Any] | None = None,
    run_kind: str = "UNVERIFIED",
    current_threshold: float = 0.6,
) -> str:
    metadata = run_payload.get("metadata", run_payload)
    case_results = metadata.get("case_results", [])
    summary = summarize_decisions(dataset, case_results)
    confidence = analyze_confidence(dataset, case_results, current_threshold)
    dims = metadata.get("dimension_summaries", [])
    thresholds = metadata.get("thresholds", {})
    dataset_info = metadata.get("dataset", {})
    lines = [
        "# Sprint 7 AI Answer Quality Scorecard",
        "",
        f"- Run ID: `{run_payload.get('id', 'not supplied')}`",
        f"- Run classification: **{run_kind}**",
        f"- Dataset: `{dataset_info.get('dataset_id', dataset.get('dataset_id'))}@{dataset_info.get('dataset_version', dataset.get('dataset_version'))}`",
        f"- Cases in dataset: **{len(dataset.get('cases', []))}**; matched evaluation results: **{summary['matched_cases']}**",
        f"- Evaluation status: `{run_payload.get('status', 'not supplied')}`",
        "",
        "## Overall evaluation",
        "",
        f"- Evaluator pass rate: **{_percent(metadata.get('overall_pass_rate'))}** ({metadata.get('passed_cases', 'N/A')} passed / {metadata.get('total_cases', summary['matched_cases'])} evaluated)",
        f"- Decision correctness from labeled outcomes: **{_percent(summary['decision_pass_rate'])}** ({summary['correct_decisions']} / {summary['matched_cases']})",
        "",
        "## Quality dimensions",
        "",
        "| Dimension | Cases | Passed | Pass rate | Mean score | Proposed minimum | Target pass rate | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in dims:
        lines.append(
            "| {dimension} | {cases} | {passed} | {rate} | {mean} | {minimum} | {target} | {decision} |".format(
                dimension=item.get("dimension", "N/A"),
                cases=item.get("evaluated_cases_count", 0),
                passed=item.get("passed_cases_count", 0),
                rate=_percent(item.get("observed_pass_rate")),
                mean=item.get("observed_mean_score", "N/A"),
                minimum=item.get("proposed_threshold_score", "N/A"),
                target=_percent(item.get("proposed_target_pass_rate")),
                decision=item.get("threshold_decision_status", thresholds.get("decision_status", "PROPOSED")),
            )
        )
    if not dims:
        lines.append("| No dimension summaries supplied | — | — | — | — | — | — | — |")

    expected = summary["expected_action_counts"]
    actual = summary["actual_decision_counts"]
    lines += [
        "",
        "## Expected actions and observed decisions",
        "",
        "| Expected action | Cases |",
        "|---|---:|",
        f"| Answer with citation | {expected[ANSWER]} |",
        f"| Safe no-answer | {expected[SAFE_NO_ANSWER]} |",
        f"| Escalate to human | {expected[ESCALATE]} |",
        "",
        "| Observed decision | Cases |",
        "|---|---:|",
    ]
    for action, count in sorted(actual.items()):
        lines.append(f"| {action} | {count} |")
    lines += [
        "",
        f"- False answers on no-answer/escalation cases: **{summary['false_answers']}**",
        f"- Unnecessary abstentions on answerable cases: **{summary['unnecessary_abstentions']}**",
        f"- Escalation cases that abstained in Python service: **{summary['escalation_abstentions']} / {summary['expected_escalations']}**",
        "- Actual ticket/escalation delivery: **NOT VERIFIED**. Python-service results do not prove that a human ticket was created or delivered.",
        "",
        "## Confidence-threshold review",
        "",
        f"- Current runtime threshold supplied to this report: **{confidence['current_runtime_threshold']:.2f}**",
        f"- Answerable confidence range: `{confidence['answerable_confidence']['min']}`–`{confidence['answerable_confidence']['max']}` (n={confidence['answerable_confidence']['count']})",
        f"- Insufficient-evidence confidence range: `{confidence['insufficient_evidence_confidence']['min']}`–`{confidence['insufficient_evidence_confidence']['max']}` (n={confidence['insufficient_evidence_confidence']['count']})",
        f"- Ranges overlap: **{confidence['confidence_ranges_overlap']}**",
        f"- Proposed threshold: **{confidence['proposed_threshold'] if confidence['proposed_threshold'] is not None else 'NONE'}**",
        f"- Recommendation: **{confidence['recommendation']}** (`{confidence['decision_status']}`)",
        "",
        confidence["limitation"],
        "Do not change `RAG_MIN_CONFIDENCE` automatically. A numerical threshold is only a candidate for review, never an approval.",
        "",
        "## ACL and latency checks (SCRUM-122)",
        "",
    ]

    acl = (operations or {}).get("acl") if operations else None
    latency = (operations or {}).get("latency") if operations else None
    if acl:
        lines += [
            f"- ACL cases: {acl.get('total_cases', 'N/A')}; passed: {acl.get('passed_cases', 'N/A')}; unauthorized data exposures: {acl.get('unauthorized_data_exposures', 'N/A')}.",
        ]
    else:
        lines.append("- ACL results: **NOT PROVIDED** (SCRUM-122).")
    if latency:
        lines += [
            f"- Latency samples: {latency.get('sample_count', 'N/A')}; p50: {latency.get('p50_ms', 'N/A')} ms; p95: {latency.get('p95_ms', 'N/A')} ms; p99: {latency.get('p99_ms', 'N/A')} ms.",
        ]
    else:
        lines.append("- Latency results: **NOT PROVIDED** (SCRUM-122).")

    lines += [
        "",
        "## Governance and limitations",
        "",
        f"- Threshold decision status: `{thresholds.get('decision_status', 'PROPOSED')}`. No threshold is approved by this report.",
        "- Escalation delivery must be verified through the Core/Java service or a dedicated end-to-end check; this report only sees Python RAG outcomes.",
        "- Dataset v1.1.0 remains English-only and primarily tests one technical source. It contains no ambiguous cases, no multi-document synthesis cases, and no conflicting-source cases.",
        f"- Run classification `{run_kind}` is an operator-supplied label; confirm the run used the intended live or offline dependencies before publication.",
        "",
    ]
    return "\n".join(lines)


def read_current_threshold(env_file: Path) -> float:
    if "RAG_MIN_CONFIDENCE" in os.environ:
        return float(os.environ["RAG_MIN_CONFIDENCE"])
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("RAG_MIN_CONFIDENCE="):
                return float(stripped.split("=", 1)[1].strip().strip('"\''))
    return 0.6


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Sprint 7 scorecard from an evaluation API response.")
    parser.add_argument("--evaluation", type=Path, required=True, help="JSON saved from GET /api/v1/rag/evaluations/runs/{run_id}")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/golden_set_v1.json"))
    parser.add_argument("--operations", type=Path, help="Optional SCRUM-122 ACL/latency JSON")
    parser.add_argument("--output", type=Path, default=Path("docs/SCRUM-34-sprint-7-scorecard.md"))
    parser.add_argument("--run-kind", choices=["LIVE", "OFFLINE", "UNVERIFIED"], default="UNVERIFIED")
    parser.add_argument("--current-threshold", type=float, help="Override RAG_MIN_CONFIDENCE read from environment/.env")
    args = parser.parse_args()

    run_payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    operations = json.loads(args.operations.read_text(encoding="utf-8")) if args.operations else None
    threshold = args.current_threshold if args.current_threshold is not None else read_current_threshold(Path(".env"))
    report = build_scorecard(run_payload, dataset, operations, args.run_kind, threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Scorecard written to {args.output}")


if __name__ == "__main__":
    main()
