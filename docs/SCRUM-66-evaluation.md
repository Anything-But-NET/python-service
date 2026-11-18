# MVP Answer-Quality Checks and Threshold Validation (`SCRUM-66`)

This document defines the four MVP answer-quality dimensions, their executable calculation methods (`SCRUM-146`), how to reproduce evaluation runs against the 60-case Golden Set (`SCRUM-163`), and the threshold validation review (`SCRUM-147`).

---

## 1. Quality Dimensions & Calculation Methods (`SCRUM-146`)

All four quality checks are implemented in `python_service.services.evaluation_service` and return a standardized `QualityCheckResult`:
- `applicable` (`bool`): whether the check applies to the case's `expected_outcome`.
- `passed` (`bool`): whether `score >= threshold` (and required structural invariants hold).
- `score` (`float` in `[0.0, 1.0]`): measurable score for the case.
- `threshold` (`float` in `[0.0, 1.0]`): proposed minimum score for the check.
- `calculation_method` (`str`): explicit identifier of the scoring formula.
- `details` (`dict`): diagnostic fields explaining why a case passed or failed.

### 1.1. Answer Relevance (`relevance`)
- **Calculation Method:** `expected_answer_and_query_token_recall_v1`
- **Applicability:** Applicable only when `expected_outcome == "ANSWER_EXPECTED"` (40 cases in `golden_set_v1.json`).
- **What is checked:**
  1. Verifies that the model did not return an empty answer or an abstention phrase when an answer was expected.
  2. Tokenizes normalized content terms (excluding stopwords) from `expected_answer` and `query` and computes token support recall against `actual_answer`:
     $$\text{score} = \max\left(R_{\text{expected}},\; 0.8 \cdot R_{\text{expected}} + 0.2 \cdot R_{\text{query}}\right)$$
- **Pass / Fail Examples:**
  - **Pass:** Query `"What tool is used for Static Code Analysis (SCA) in Stage 3?"`, Expected `"SonarQube"`, Actual `"Stage 3 uses SonarQube for Static Code Analysis."` $\rightarrow$ `score = 1.0 >= 0.75` (`passed = true`).
  - **Fail:** Actual `"I couldn't find information about this in the knowledge base."` $\rightarrow$ `score = 0.0 < 0.75` (`passed = false`).

### 1.2. Factual Grounding (`factual_grounding`)
- **Calculation Method:** `retrieved_passage_claim_support_ratio_v1`
- **Applicability:** Applicable when `expected_outcome == "ANSWER_EXPECTED"`.
- **What is checked:**
  1. Splits `actual_answer` into individual sentence claims.
  2. Checks each claim against the concatenated text of `retrieved_passages`. A claim is marked as supported if at least `60%` of its normalized content tokens appear in the retrieved evidence.
  3. Computes the ratio of supported claims to total claims:
     $$\text{score} = \frac{\text{supported\_claims\_count}}{\text{total\_claims\_count}}$$
  4. Records any unverified statements in `details.unsupported_claims`.
- **Pass / Fail Examples:**
  - **Pass:** Retrieved passage states `"Artifacts are stored in Amazon ECR with AES-256 encryption."`; Actual answer states `"Artifacts are stored in Amazon ECR."` $\rightarrow$ `1/1` claims supported (`score = 1.0 >= 0.85`, `passed = true`).
  - **Fail:** Actual answer adds an ungrounded sentence `"It also replicates images to Azure Container Registry."` $\rightarrow$ `1/2` claims supported (`score = 0.50 < 0.85`, `passed = false`, `unsupported_claims` populated).

### 1.3. Citation Correctness (`citation_correctness`)
- **Calculation Method:** `citation_existence_source_and_support_v1`
- **Applicability:** Applicable when `expected_outcome == "ANSWER_EXPECTED"`. (For `INSUFFICIENT_EVIDENCE`, it verifies that no hallucinated citations were returned).
- **What is checked:** Evaluates three mandatory criteria (`score = passed_subchecks / 3.0`):
  1. `citations_exist`: `len(actual_citations) > 0`.
  2. `resolves_to_expected_source`: all cited `document_id` values match the case's `expected_document_ids` (no unexpected documents cited).
  3. `supports_answer_content`: every cited `chunk_id` belongs to the retrieved passage set and the cited chunk text supports at least `50%` of the answer's content tokens.
- **Pass / Fail Examples:**
  - **Pass:** Cited `document_id` is `"eng-backend-deployment-pipeline-architecture"`, `chunk_id` is in retrieved passages, and chunk text contains the answer facts $\rightarrow$ `3/3` sub-checks pass (`score = 1.0`, `passed = true`).
  - **Fail:** Model cites `wrong-doc-id` or omits citations $\rightarrow$ `resolves_to_expected_source = false` (`score < 1.0`, `passed = false`).

### 1.4. Safe No-Answer Behavior (`safe_no_answer`)
- **Calculation Method:** `abstention_status_and_hallucination_guard_v1`
- **Applicability:** Applicable when `expected_outcome == "INSUFFICIENT_EVIDENCE"` (20 unanswerable cases in `golden_set_v1.json`: 10 off-topic/spam + 10 missing-documentation queries).
- **What is checked:**
  - For `INSUFFICIENT_EVIDENCE`: passes (`score = 1.0`) if and only if the pipeline abstains safely (`actual_status in ("EMPTY_RETRIEVAL", "LOW_CONFIDENCE")` or explicit refusal text) with zero citations (`len(actual_citations) == 0`) and `actual_status != "ERROR"`.
  - For `ANSWER_EXPECTED`: tracks `details.unnecessary_abstention` so the team can distinguish unnecessary abstentions from factual errors.
- **Pass / Fail Examples:**
  - **Pass:** Query `"What is the Wi-Fi password for the guest network?"` $\rightarrow$ `status = "EMPTY_RETRIEVAL"`, `answer = null`, `citations = []` $\rightarrow$ `score = 1.0`, `passed = true`.
  - **Fail:** Model fabricates a Wi-Fi password with `status = "ANSWER_GENERATED"` $\rightarrow$ `score = 0.0`, `passed = false`.

---

## 2. Threshold Proposals & Golden Set Validation (`SCRUM-147`)

> **Important Governance Note:** All thresholds below carry `decision_status: "PROPOSED"`. They are proposals for technical lead / product review and must not be silently treated as approved product decisions until explicitly signed off in `SCRUM-34`.

### 2.1. Proposed Thresholds vs. Golden Set Baseline

Dataset: `datasets/golden_set_v1.json` (`dataset_id: "mvp-golden-set"`, `dataset_version: "v1.0.0"`, 60 cases from `SCRUM-10`).

| Dimension | Applicable Cases | Proposed Case Min Score | Proposed Target Pass Rate | Decision Status | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `relevance` | 40 (`ANSWER_EXPECTED`) | `0.75` | `0.85` (85%) | `PROPOSED` | Checks factual overlap with expected Golden Set answer |
| `factual_grounding` | 40 (`ANSWER_EXPECTED`) | `0.85` | `0.85` (85%) | `PROPOSED` | Aligned with `SCRUM-34` proposed faithfulness target ($\ge 85\%$) |
| `citation_correctness` | 40 (`ANSWER_EXPECTED`) | `1.00` | `0.85` (85%) | `PROPOSED` | Requires all 3 citation sub-checks to pass per case |
| `safe_no_answer` | 20 (`INSUFFICIENT_EVIDENCE`) | `1.00` | `0.85` (85%) | `PROPOSED` | Binary check (`1.0` on safe abstention, `0.0` on hallucination) |

### 2.2. Known Limitations of Golden Set `v1.0.0`
1. **Single-document coverage for answerable queries:** All 40 answerable cases (`GS-021` to `GS-060`) reference a single technical specification (`[Eng] Backend Deployment Pipeline Architecture`). Multi-document retrieval and cross-document citation disambiguation are not yet stressed.
2. **No conflicting-source cases:** The current dataset does not contain contradictory passages across different documents to test conflict abstention.
3. **English-only queries:** All 60 questions are in English; multilingual (Ukrainian) query handling is not covered in `v1.0.0`.
4. **Placeholder token formatting (`GS-039`, `GS-041`, `GS-044`, `GS-045`):** Four expected answers in `SCRUM-10` are bracketed placeholders (`[DB_PROD_CLUSTER]`, `[REDIS_PROD_CLUSTER]`, `[EKS_ENDPOINT]`, `[ALB_ENDPOINT]`), which require exact token matching from the ingested markdown fixture.

---

## 3. Reproducing an Evaluation Run (`SCRUM-163`)

### 3.1. Automated Test Suite Execution
Run the repeatable unit and 60-case Golden Set evaluation tests via `pytest`:

```bash
uv run pytest tests/test_evaluation_service.py
```

### 3.2. Triggering an Evaluation Run via REST API
Contracts are defined in `environment/contracts/openapi.yaml` (`POST /api/v1/rag/evaluations/runs` and `GET /api/v1/rag/evaluations/runs/{run_id}`).

```bash
curl -X POST http://localhost:8000/api/v1/rag/evaluations/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_version_id": "00000000-0000-0000-0000-000000000001",
    "dataset": {
      "dataset_id": "mvp-golden-set",
      "dataset_version": "v1.0.0",
      "dataset_path": "datasets/golden_set_v1.json"
    },
    "thresholds": {
      "decision_status": "PROPOSED",
      "relevance_min_score": 0.75,
      "factual_grounding_min_score": 0.85,
      "citation_correctness_min_score": 1.0,
      "target_pass_rate": 0.85
    }
  }'
```

To fetch a persisted evaluation report by `run_id`:

```bash
curl http://localhost:8000/api/v1/rag/evaluations/runs/<run_id>
```
