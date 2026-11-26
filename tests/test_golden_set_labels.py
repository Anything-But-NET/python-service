import json
from collections import Counter
from pathlib import Path


def test_golden_set_labels():
    path = Path(__file__).resolve().parents[1] / "datasets/golden_set_v1_1.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))

    assert dataset["dataset_version"] == "v1.1.0"
    assert len(dataset["cases"]) == 63

    cases = dataset["cases"]

    assert Counter(c["expected_action"] for c in cases) == {
        "ANSWER_WITH_CITATION": 40,
        "SAFE_NO_ANSWER": 13,
        "ESCALATE_TO_HUMAN": 10,
    }

    assert Counter(c["question_scope"] for c in cases) == {
        "IN_SCOPE": 50,
        "AMBIGUOUS": 3,
        "OUT_OF_SCOPE": 8,
        "RESTRICTED": 2,
    }

    assert Counter(c["source_availability"] for c in cases) == {
        "AVAILABLE_IN_EVALUATED_CORPUS": 43,
        "NOT_FOUND_IN_EVALUATED_CORPUS": 10,
        "NOT_APPLICABLE": 10,
    }

    ambiguous = [c for c in cases if c["question_scope"] == "AMBIGUOUS"]
    assert {c["case_id"] for c in ambiguous} == {"GS-061", "GS-062", "GS-063"}
    assert all(c["expected_outcome"] == "INSUFFICIENT_EVIDENCE" for c in ambiguous)
    assert all(c["expected_action"] == "SAFE_NO_ANSWER" for c in ambiguous)
    assert all("clarification" in c["expected_behavior"].lower() for c in ambiguous)
    assert all(c["expected_outcome"] for c in cases)
