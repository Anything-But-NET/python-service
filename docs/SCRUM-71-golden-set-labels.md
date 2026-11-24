# SCRUM-71 — Golden Set expected-outcome labels

## Dataset revision

The labeled dataset is `datasets/golden_set_v1.json`, now version `v1.1.0`. The 60 original questions and answer expectations are preserved; the version increment adds structured labels without changing the old `expected_outcome`, `expected_behavior`, source fields, or permissions.

## Labels

| Field | Values | Meaning |
|---|---|---|
| `expected_outcome` | `ANSWER_EXPECTED`, `INSUFFICIENT_EVIDENCE` | Existing evaluator answerability label; retained for backward compatibility |
| `expected_action` | `ANSWER_WITH_CITATION`, `SAFE_NO_ANSWER`, `ESCALATE_TO_HUMAN` | Required system action |
| `question_scope` | `IN_SCOPE`, `OUT_OF_SCOPE`, `RESTRICTED`, `AMBIGUOUS` | Scope/policy classification |
| `source_availability` | `AVAILABLE_IN_EVALUATED_CORPUS`, `NOT_FOUND_IN_EVALUATED_CORPUS`, `NOT_APPLICABLE` | Whether the expected source is available in the corpus used for evaluation |

## v1.1.0 distribution

| Label | Distribution |
|---|---:|
| Answer with citation | 40 |
| Safe no-answer | 10 |
| Escalate to human | 10 |
| In scope | 50 |
| Out of scope | 8 |
| Restricted | 2 |
| Ambiguous | 0 |

There are currently no examples labeled `AMBIGUOUS`. This is an explicit dataset coverage gap; the label is supported by the schema but no synthetic ambiguous examples were added. Do not report ambiguous-case coverage until the product/rubric owner supplies representative cases.

`GS-002` (CEO compensation) and `GS-005` (personal phone number) are labeled `RESTRICTED`, not merely out-of-scope. The ten work-related questions without a source in the evaluated corpus are marked in-scope and expected to escalate. The 40 source-backed technical questions remain in-scope and require a cited answer.

## Compatibility

`expected_outcome` remains unchanged so the existing SCRUM-66 evaluator continues to load the dataset. The scorecard tool in `scripts/generate_sprint7_scorecard.py` joins evaluation results to the new labels by `case_id` and calculates action-level outcomes separately from the four existing quality dimensions.
