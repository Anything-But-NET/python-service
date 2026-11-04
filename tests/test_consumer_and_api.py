import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from python_service.api.routes import _to_search_result
from python_service.kafka.consumer import IngestionKafkaWorker
from python_service.models.events import DocumentDeleteEvent


@pytest.fixture
def worker():
    with (
        patch("python_service.kafka.consumer.Consumer"),
        patch("python_service.kafka.consumer.Producer"),
    ):
        w = IngestionKafkaWorker(
            parser=MagicMock(),
            vectorizer=MagicMock(),
            db=AsyncMock(),
            topic="rag.documents.ingestion.v1",
        )
    return w


@pytest.mark.asyncio
async def test_process_message_payload_dispatches_event(worker):
    worker._handle_event = AsyncMock()
    raw = json.dumps({
        "sync_id": str(uuid.uuid4()),
        "document_id": "doc-1",
        "action": "delete",
    }).encode("utf-8")

    await worker.process_message_payload(raw)

    event = worker._handle_event.await_args.args[0]
    assert isinstance(event, DocumentDeleteEvent)
    assert event.document_id == "doc-1"


@pytest.mark.asyncio
async def test_process_message_payload_rejects_invalid_payloads(worker):
    worker._handle_event = AsyncMock()

    with pytest.raises(json.JSONDecodeError):
        await worker.process_message_payload(b"not json")
    with pytest.raises(ValidationError):
        await worker.process_message_payload(b'{"document_id": "d", "action": "nope"}')

    worker._handle_event.assert_not_awaited()


def test_send_to_dlt_preserves_message_and_adds_headers(worker):
    msg = MagicMock()
    msg.headers.return_value = None
    msg.topic.return_value = "rag.documents.ingestion.v1"
    msg.partition.return_value = 0
    msg.offset.return_value = 42
    msg.key.return_value = b"doc-1"
    msg.value.return_value = b"payload"

    worker._send_to_dlt(msg, ValueError("boom"))

    kwargs = worker.producer.produce.call_args.kwargs
    assert kwargs["topic"] == "rag.documents.ingestion.v1-dlt"
    assert kwargs["value"] == b"payload"
    headers = dict(kwargs["headers"])
    assert headers["error_reason"] == b"boom"
    assert headers["original_offset"] == b"42"


def test_search_result_citations_follow_rag_citation_contract():
    chunk_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    item = _to_search_result({
        "point_id": str(uuid.uuid4()),
        "parent_id": None,
        "score": 0.91,
        "children_count": 2,
        "chunk_ids": chunk_ids,
        "chunks": ["a", "b"],
        "document_id": "doc-a",
        "text": "a\nb",
    })

    citations = [c.model_dump() for c in item.citations]
    assert citations == [
        {"document_id": "doc-a", "chunk_id": chunk_ids[0], "score": 0.91},
        {"document_id": "doc-a", "chunk_id": chunk_ids[1], "score": 0.91},
    ]
