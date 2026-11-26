# Golden Set v1.1.0 labels (SCRUM-71)

## Dataset versions

- `datasets/golden_set_v1.json` remains the unchanged **v1.0.0** dataset for existing evaluator requests.
- `datasets/golden_set_v1_1.json` is **v1.1.0**, containing the SCRUM-71 labels and three explicit ambiguous cases.
- When an evaluation request omits `dataset_path`, `EvaluationService.load_golden_set` selects the file by `dataset_version`. An explicit path must point to a file whose metadata matches the requested version.

## Labels

- `expected_action`: `ANSWER_WITH_CITATION`, `SAFE_NO_ANSWER`, or `ESCALATE_TO_HUMAN`.
- `question_scope`: `IN_SCOPE`, `AMBIGUOUS`, `OUT_OF_SCOPE`, or `RESTRICTED`.
- `source_availability`: `AVAILABLE_IN_EVALUATED_CORPUS`, `NOT_FOUND_IN_EVALUATED_CORPUS`, or `NOT_APPLICABLE`.
- `expected_outcome` remains unchanged for compatibility with the existing answer-quality evaluator.

An `AMBIGUOUS` case is related to the evaluated technical knowledge, but it lacks a discriminator needed to select one supported answer. The expected behavior is to ask a clarifying question, not guess and not create a human-escalation ticket. For evaluator compatibility, these cases use `expected_outcome=INSUFFICIENT_EVIDENCE` and `expected_action=SAFE_NO_ANSWER`.

## v1.1.0 coverage

There are 63 cases:

| Dimension | Label | Cases |
|---|---|---:|
| Expected action | `ANSWER_WITH_CITATION` | 40 |
| Expected action | `SAFE_NO_ANSWER` | 13 |
| Expected action | `ESCALATE_TO_HUMAN` | 10 |
| Question scope | `IN_SCOPE` | 50 |
| Question scope | `AMBIGUOUS` | 3 |
| Question scope | `OUT_OF_SCOPE` | 8 |
| Question scope | `RESTRICTED` | 2 |
| Source availability | `AVAILABLE_IN_EVALUATED_CORPUS` | 43 |
| Source availability | `NOT_FOUND_IN_EVALUATED_CORPUS` | 10 |
| Source availability | `NOT_APPLICABLE` | 10 |

Ambiguous examples:

- **GS-061:** “Which production cluster endpoint should I use?” — clarify whether this means the database or Redis cluster.
- **GS-062:** “What endpoint should the backend call?” — clarify which backend component or dependency is meant.
- **GS-063:** “Which Helm configuration values should I use?” — clarify whether the target is Staging or Production.

The 40 answerable cases still target the single technical document used by the original Golden Set. Multi-document synthesis, conflicting-source handling, and multilingual query coverage remain limitations.
