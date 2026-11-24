import json
from collections import Counter
from pathlib import Path


def test_golden_set_labels():
    path = Path(__file__).resolve().parents[1] / "datasets/golden_set_v1.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))

    assert dataset["dataset_version"] == "v1.1.0"
    assert len(dataset["cases"]) == 60

    cases = dataset["cases"]

    assert Counter(c["expected_action"] for c in cases) == {
        "ANSWER_WITH_CITATION": 40,
        "SAFE_NO_ANSWER": 10,
        "ESCALATE_TO_HUMAN": 10,
    }

    assert Counter(c["question_scope"] for c in cases) == {
        "IN_SCOPE": 50,
        "OUT_OF_SCOPE": 8,
        "RESTRICTED": 2,
    }

    assert Counter(c["source_availability"] for c in cases) == {
        "AVAILABLE_IN_EVALUATED_CORPUS": 40,
        "NOT_FOUND_IN_EVALUATED_CORPUS": 10,
        "NOT_APPLICABLE": 10,
    }

    assert all(c["expected_outcome"] for c in cases)
