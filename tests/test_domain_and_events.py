import uuid

import pytest
from pydantic import ValidationError

from python_service.config import Settings, get_settings
from python_service.models.domain import SegmentNode, VectorizedSegmentNode
from python_service.models.events import (
    DocumentDeleteEvent,
    DocumentUpdateAclEvent,
    DocumentUpsertEvent,
    event_adapter,
)


def test_segment_node_add_content():
    node = SegmentNode(name="Header", level=1)
    assert node.content == ""

    node.add_content("First line")
    assert node.content == "First line"

    node.add_content("Second line")
    assert node.content == "First line\nSecond line"


def test_vectorized_segment_node():
    node = VectorizedSegmentNode(
        name="Root",
        level=0,
        embedding=None,
        content=[("text", None)],
    )
    assert node.name == "Root"
    assert node.level == 0
    assert node.embedding is None
    assert len(node.content) == 1
    assert node.children == []


def test_event_adapter_upsert_validation():
    payload = {
        "sync_id": str(uuid.uuid4()),
        "document_id": "doc-123",
        "action": "create",
        "title": "Test Title",
        "raw_content": "# Hello",
        "acl": {"allowed_groups": ["group-1"]},
    }
    event = event_adapter.validate_python(payload)
    assert isinstance(event, DocumentUpsertEvent)
    assert event.document_id == "doc-123"
    assert event.action == "create"
    assert event.acl.allowed_groups == ["group-1"]


def test_event_adapter_delete_validation():
    payload = {
        "document_id": "doc-456",
        "action": "delete",
    }
    event = event_adapter.validate_python(payload)
    assert isinstance(event, DocumentDeleteEvent)
    assert event.document_id == "doc-456"
    assert event.action == "delete"


def test_event_adapter_update_acl_validation():
    payload = {
        "document_id": "doc-789",
        "action": "update_acl",
        "acl": {"allowed_groups": ["admin"]},
    }
    event = event_adapter.validate_python(payload)
    assert isinstance(event, DocumentUpdateAclEvent)
    assert event.document_id == "doc-789"
    assert event.acl.allowed_groups == ["admin"]


def test_event_adapter_invalid_action():
    payload = {
        "document_id": "doc-invalid",
        "action": "unknown_action",
    }
    with pytest.raises(ValidationError):
        event_adapter.validate_python(payload)


def test_settings_and_aliases():
    settings = Settings(
        DB_URL="postgresql+psycopg://test:test@localhost:5432/test",
        KAFKA_BROKERS="broker1:9092,broker2:9092",
        KAFKA_TOPIC="custom.topic",
        KAFKA_GROUP_ID="custom-group",
        KAFKA_ENABLED=False,
        VECTORIZER_GRPC_TARGET="vectorizer:50051",
        API_HOST="127.0.0.1",
        API_PORT=9000,
    )
    assert settings.db_url == "postgresql+psycopg://test:test@localhost:5432/test"
    assert settings.kafka_brokers == "broker1:9092,broker2:9092"
    assert settings.kafka_topic == "custom.topic"
    assert settings.kafka_group_id == "custom-group"
    assert settings.kafka_enabled is False
    assert settings.vectorizer_grpc_target == "vectorizer:50051"
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 9000

    # Also verify get_settings() returns a cached Settings instance
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
