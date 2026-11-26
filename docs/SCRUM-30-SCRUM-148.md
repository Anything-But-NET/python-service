# Grounded answers and retrieved-content safety

## Runtime flow

`POST /api/v1/rag/generate` is the generation contract consumed by Java Core. Python embeds the question, retrieves passages through the existing ACL-filtered RAG query, and sends only those passages to an OpenAI-compatible chat-completions endpoint. Java remains responsible for resolving `document_id` to display metadata (`title` and `url`).

The response uses the Java contract:

- `query_id`
- `status`: `ANSWER_GENERATED`, `EMPTY_RETRIEVAL`, `LOW_CONFIDENCE`, or `ERROR`
- `answer`
- `confidence_score`
- `citations`: `document_id`, `chunk_id`, `score`

An answer is returned only when retrieval has evidence above the configured relevance threshold, the model reports sufficient evidence and acceptable confidence, and every cited chunk ID belongs to the retrieved set. Otherwise the service returns a non-answer status and empty citations so Java can escalate safely.

## Prompt-injection handling

The immutable security and answer policy is sent in a single `system` message. A non-default, versioned prompt is JSON-serialized as lower-priority task guidance within that same `system` message, followed by an immutable security reminder; it never relies on a provider-specific `developer` role and cannot override the security policy or required JSON schema. The question and retrieved passages are JSON-serialized in a separate `user` message; retrieved passage text is untrusted reference data, never instructions. Passwords, API keys, access tokens, private keys, and other credentials from retrieved passages must not be disclosed. JSON serialization preserves quotes, newlines, and embedded JSON as string values, but it is not a guarantee against model-level prompt injection. No tools are exposed to the model. The response parser never executes model output and rejects citations that were not part of the retrieved evidence.

This is defense-in-depth at the prompt/data boundary, not a hard isolation boundary or a guarantee against model-level prompt injection. It does not replace provider-side safety controls or the SCRUM-149 adversarial test suite.

## Configuration

Set these values in the runtime environment (see `.env.example`):

- `LLM_BASE_URL`: OpenAI-compatible API base URL, default `https://api.openai.com/v1`.
- `LLM_API_KEY`: provider secret; do not commit it.
- `LLM_MODEL`: deployed model name, default `gpt-4o-mini`.
- `LLM_TIMEOUT_SECONDS`, `LLM_MAX_TOKENS`: provider request limits.
- `RAG_RETRIEVAL_LIMIT`, `RAG_AUTO_MERGE_THRESHOLD`: retrieval settings.
- `RAG_MIN_RELEVANCE_SCORE`, `RAG_MIN_CONFIDENCE`: conservative answer gates. Tune these with representative evaluation data; the returned confidence is a conservative heuristic, not a statistically calibrated probability.
- `RAG_MAX_CONTEXT_CHARS`: upper bound on retrieved text sent to the model.

## Manual smoke test

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "query_id": "smoke-1",
    "query": "How do I request annual leave?",
    "user_permissions": {"allowed_groups": ["employees"]}
  }'
```
