import json
import re
from pathlib import Path

from scripts.generate_sprint7_scorecard import analyze_confidence, summarize_decisions


ROOT = Path(__file__).resolve().parents[1]


def test_golden_set_v11_has_explicit_outcome_labels():
    dataset = json.loads((ROOT / "datasets/golden_set_v1.json").read_text(encoding="utf-8"))
    assert dataset["dataset_version"] == "v1.1.0"
    assert len(dataset["cases"]) == 60

    actions = {}
    scopes = {}
    sources = {}
    for case in dataset["cases"]:
        actions[case["expected_action"]] = actions.get(case["expected_action"], 0) + 1
        scopes[case["question_scope"]] = scopes.get(case["question_scope"], 0) + 1
        sources[case["source_availability"]] = sources.get(case["source_availability"], 0) + 1

    assert actions == {
        "SAFE_NO_ANSWER": 10,
        "ESCALATE_TO_HUMAN": 10,
        "ANSWER_WITH_CITATION": 40,
    }
    assert scopes == {"OUT_OF_SCOPE": 8, "RESTRICTED": 2, "IN_SCOPE": 50}
    assert sources == {
        "NOT_APPLICABLE": 10,
        "NOT_FOUND_IN_EVALUATED_CORPUS": 10,
        "AVAILABLE_IN_EVALUATED_CORPUS": 40,
    }
    assert "AMBIGUOUS" not in scopes


def test_decision_summary_counts_false_answers_and_keeps_escalation_unverified():
    dataset = {
        "cases": [
            {"case_id": "GS-021", "expected_action": "ANSWER_WITH_CITATION"},
            {"case_id": "GS-001", "expected_action": "SAFE_NO_ANSWER"},
            {"case_id": "GS-011", "expected_action": "ESCALATE_TO_HUMAN"},
            {"case_id": "GS-012", "expected_action": "ESCALATE_TO_HUMAN"},
        ]
    }
    results = [
        {"case_id": "GS-021", "actual_status": "ANSWER_GENERATED", "actual_answer": "Docker", "actual_citations": [{"document_id": "doc"}]},
        {"case_id": "GS-001", "actual_status": "EMPTY_RETRIEVAL", "actual_answer": None, "actual_citations": []},
        {"case_id": "GS-011", "actual_status": "LOW_CONFIDENCE", "actual_answer": None, "actual_citations": []},
        {"case_id": "GS-012", "actual_status": "ANSWER_GENERATED", "actual_answer": "Made-up answer", "actual_citations": [{"document_id": "doc"}]},
    ]

    summary = summarize_decisions(dataset, results)
    assert summary["matched_cases"] == 4
    assert summary["correct_decisions"] == 3
    assert summary["false_answers"] == 1
    assert summary["escalation_abstentions"] == 1
    assert summary["escalation_delivery_verification"] == "NOT_VERIFIABLE_IN_PYTHON_SERVICE"


def test_scorecard_marks_missing_acl_latency_and_ticket_delivery_as_unverified():
    dataset = {
        "dataset_id": "mvp-golden-set",
        "dataset_version": "v1.1.0",
        "cases": [{"case_id": "GS-021", "expected_action": "ANSWER_WITH_CITATION"}],
    }
    evaluation = {
        "id": "run-1",
        "status": "completed",
        "metadata": {
            "dataset": {"dataset_id": "mvp-golden-set", "dataset_version": "v1.1.0"},
            "total_cases": 1,
            "passed_cases": 1,
            "overall_pass_rate": 1.0,
            "case_results": [{
                "case_id": "GS-021",
                "actual_status": "ANSWER_GENERATED",
                "actual_answer": "Docker",
                "actual_citations": [{"document_id": "doc"}],
                "actual_confidence_score": 0.9,
            }],
            "dimension_summaries": [],
            "thresholds": {"decision_status": "PROPOSED"},
        },
    }
    from scripts.generate_sprint7_scorecard import build_scorecard

    report = build_scorecard(evaluation, dataset)
    assert "ACL results: **NOT PROVIDED**" in report
    assert "Latency results: **NOT PROVIDED**" in report
    assert "Actual ticket/escalation delivery: **NOT VERIFIED**" in report


def test_confidence_analysis_only_proposes_threshold_when_ranges_do_not_overlap():
    dataset = {
        "cases": [
            {"case_id": "GS-021", "expected_action": "ANSWER_WITH_CITATION"},
            {"case_id": "GS-022", "expected_action": "ANSWER_WITH_CITATION"},
            {"case_id": "GS-001", "expected_action": "SAFE_NO_ANSWER"},
        ]
    }
    separated_results = [
        {"case_id": "GS-021", "actual_confidence_score": 0.8},
        {"case_id": "GS-022", "actual_confidence_score": 0.9},
        {"case_id": "GS-001", "actual_confidence_score": 0.4},
    ]
    separated = analyze_confidence(dataset, separated_results, 0.6)
    assert separated["confidence_ranges_overlap"] is False
    assert separated["proposed_threshold"] == 0.6
    assert separated["decision_status"] == "PROPOSED"

    overlapping_results = separated_results[:-1] + [
        {"case_id": "GS-001", "actual_confidence_score": 0.85}
    ]
    overlapping = analyze_confidence(dataset, overlapping_results, 0.6)
    assert overlapping["confidence_ranges_overlap"] is True
    assert overlapping["proposed_threshold"] is None
    assert overlapping["recommendation"] == "NO_SEPARATING_THRESHOLD"
