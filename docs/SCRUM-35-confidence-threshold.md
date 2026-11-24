# SCRUM-35 — Confidence threshold and escalation review

## Evaluation method

Use Golden Set `v1.1.0` with the same prompt version used for the quality evaluation. Compare the observed `actual_confidence_score` distributions for:

- 40 `ANSWER_WITH_CITATION` cases;
- 10 `SAFE_NO_ANSWER` cases;
- 10 `ESCALATE_TO_HUMAN` cases.

The analysis reports the min/max/mean confidence for answerable and insufficient-evidence cases. It proposes a separating threshold only when the observed ranges do not overlap. If they overlap, there is no defensible single confidence cutoff in this sample; preserve the current runtime setting and investigate retrieval/query-filter failures rather than selecting an arbitrary number.

## Decision interpretation

`ANSWER_GENERATED` with citations is the expected answer action. `EMPTY_RETRIEVAL` and `LOW_CONFIDENCE` are observed abstention outcomes. Answering a safe-no-answer or escalation case is counted as a false answer; abstaining on a source-backed answerable case is counted as an unnecessary abstention.

The Python RAG response does not prove that a human ticket was actually created or delivered. For `ESCALATE_TO_HUMAN`, this analysis can confirm that the RAG service abstained and exposed an escalation candidate, but ticket delivery must be verified through Core/Java or an end-to-end escalation test (SCRUM-79). That verification is deliberately not inferred from a Python status.

## Threshold governance

The configured default `RAG_MIN_CONFIDENCE` is `0.60`. The analysis reads the current value from process environment or `.env` (falling back to the code default). It does not change configuration. Any suggested value is marked `PROPOSED` and requires explicit technical/product approval before runtime adoption.

The confidence-range method is a distribution diagnostic, not a replay of model generation under different thresholds. A larger, multi-document and multilingual evaluation set is required before treating a candidate as a robust production threshold.

## Reproduction

After running the evaluator and saving the GET response for the evaluation run as JSON:

```bash
uv run python scripts/generate_sprint7_scorecard.py \
  --evaluation reports/sprint7-evaluation-run.json \
  --run-kind LIVE
```

Use `--run-kind OFFLINE` for mocked/simulated results. Supply `--current-threshold` when the evaluation run used a threshold different from the current `.env` configuration.
