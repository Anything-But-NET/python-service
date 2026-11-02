import pytest
import json
from unittest.mock import MagicMock
from python_service.services.ingestion_processor import (
    IngestionProcessor,
    RetryDLQStore,
    logger,
)
@pytest.fixture
def setup_processor():
    retry_store = RetryDLQStore()
    processor = IngestionProcessor(retry_store=retry_store)
    return processor, retry_store


@pytest.fixture
def valid_create_event():
    return {
        "document_id": "sp-doc-98412",
        "action": "create",
        "source_system": "sharepoint",
        "title": "IT Compensation Guidelines",
        "url": "https://company.sharepoint.com/docs/it-comp",
        "raw_content": "Секретний текст документа про компенсації...",
        "last_updated": "2026-10-21T14:30:00Z",
        "acl": {"allowed_groups": ["devs", "all-employees"]},
    }


def test_successful_ingestion_logging(setup_processor, valid_create_event, caplog):
    processor, _ = setup_processor

    with caplog.at_level("INFO"):
        result = processor.process_event(valid_create_event, run_id="run-123")

    # 1. Перевірка статусу результату
    assert result.status == "SUCCESS"
    assert result.document_id == "sp-doc-98412"
    assert result.run_id == "run-123"

    # 2. Перевірка відсутності секретів та raw_content у логах
    log_text = caplog.text
    assert "INGESTION_SUCCESS" in log_text
    assert "sp-doc-98412" in log_text
    assert "run-123" in log_text
    assert "Секретний текст документа" not in log_text


def test_failed_ingestion_missing_required_fields(setup_processor, caplog):
    processor, retry_store = setup_processor

    # Подія без обов'язкового url та title для дії create
    invalid_event = {
        "document_id": "sp-doc-err-01",
        "action": "create",
        "source_system": "sharepoint",
        "raw_content": "Текст...",
        "last_updated": "2026-10-21T14:30:00Z",
    }

    with caplog.at_level("ERROR"):
        result = processor.process_event(invalid_event, run_id="run-failed-01")

    # 1. Перевірка статусу та фіксація контексту запуску
    assert result.status == "FAILURE"
    assert result.document_id == "sp-doc-err-01"
    assert result.run_id == "run-failed-01"
    assert result.error_message is not None

    # 2. Перевірка логів
    log_text = caplog.text
    assert "INGESTION_FAILURE" in log_text
    assert "sp-doc-err-01" in log_text

    # 3. Перевірка перенаправлення для повторної обробки (Retry/DLQ)
    failed_record = retry_store.get_failed_document("sp-doc-err-01")
    assert failed_record is not None
    assert failed_record["document_id"] == "sp-doc-err-01"
    assert failed_record["run_id"] == "run-failed-01"
    assert "raw_content" not in failed_record["payload"]


def test_failed_ingestion_invalid_enum_format(setup_processor):
    processor, retry_store = setup_processor

    invalid_enum_event = {
        "document_id": "sp-doc-err-02",
        "action": "invalid_action_name",
        "source_system": "sharepoint",
        "last_updated": "2026-10-21T14:30:00Z",
    }

    result = processor.process_event(invalid_enum_event, run_id="run-failed-02")

    assert result.status == "FAILURE"
    assert result.document_id == "sp-doc-err-02"

    failed_record = retry_store.get_failed_document("sp-doc-err-02")
    assert failed_record is not None
    assert failed_record["run_id"] == "run-failed-02"