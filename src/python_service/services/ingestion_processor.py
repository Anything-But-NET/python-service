import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl

# Налаштування структурованого JSON-логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion_service")


class ActionEnum(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    UPDATE_ACL = "update_acl"


class SourceSystemEnum(str, Enum):
    SHAREPOINT = "sharepoint"
    M365_DOCS = "m365_docs"
    DOCUMENT_PROVIDER = "document_provider"
    RESOLVED_TICKETS = "resolved_tickets"


class AclPayload(BaseModel):
    allowed_groups: List[str] = Field(..., min_length=1)


class KafkaNewFileDataEvent(BaseModel):
    document_id: str = Field(..., min_length=1)
    action: ActionEnum
    source_system: SourceSystemEnum
    title: Optional[str] = Field(None, min_length=1)
    url: Optional[HttpUrl] = None
    raw_content: Optional[str] = Field(None, min_length=1)
    last_updated: datetime
    acl: Optional[AclPayload] = None


class IngestionResult(BaseModel):
    run_id: str
    document_id: str
    status: str  # "SUCCESS" або "FAILURE"
    source_system: Optional[str] = None
    action: Optional[str] = None
    chunks_count: int = 0
    error_message: Optional[str] = None
    timestamp: str


class RetryDLQStore:
    """Сховище або черга для ідентифікації проблемних документів під час повторної обробки (Retry)."""

    def __init__(self):
        self.failed_records: Dict[str, Dict[str, Any]] = {}

    def add_failed_document(
        self, document_id: str, run_id: str, error_reason: str, raw_event: Dict[str, Any]
    ):
        # Зберігаємо лише необхідний контекст для повтору без витоку секретів
        sanitized_event = {k: v for k, v in raw_event.items() if k != "raw_content"}
        self.failed_records[document_id] = {
            "document_id": document_id,
            "run_id": run_id,
            "error_reason": error_reason,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "payload": sanitized_event,
        }

    def get_failed_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self.failed_records.get(document_id)


class IngestionProcessor:
    def __init__(self, retry_store: RetryDLQStore):
        self.retry_store = retry_store

    def _sanitize_for_logging(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Видаляє raw_content та можливі секрети перед записом у лог."""
        sanitized = payload.copy()
        sanitized.pop("raw_content", None)
        sanitized.pop("token", None)
        sanitized.pop("secret", None)
        return sanitized

    def _log_outcome(self, result: IngestionResult):
        """Фіксує результат виконання у логах у форматі JSON без секретів та вмісту документа."""
        log_data = result.model_dump(exclude_none=True)
        if result.status == "SUCCESS":
            logger.info(f"INGESTION_SUCCESS: {json.dumps(log_data)}")
        else:
            logger.error(f"INGESTION_FAILURE: {json.dumps(log_data)}")

    def process_event(self, raw_event: Dict[str, Any], run_id: Optional[str] = None) -> IngestionResult:
        current_run_id = run_id or str(uuid.uuid4())
        doc_id = raw_event.get("document_id", "UNKNOWN_DOC_ID")
        timestamp_now = datetime.now(timezone.utc).isoformat()

        try:
            # 1. Валідація схеми контракту події
            event = KafkaNewFileDataEvent(**raw_event)

            # 2. Перевірка обов'язковості полів відповідно до специфікації контрактів (SCRUM-87)
            if event.action in [ActionEnum.CREATE, ActionEnum.UPDATE]:
                if not event.title or not event.url or not event.raw_content or not event.acl:
                    raise ValueError(
                        f"Fields 'title', 'url', 'raw_content', and 'acl' are required for action '{event.action.value}'"
                    )

            if event.action == ActionEnum.UPDATE_ACL:
                if not event.acl:
                    raise ValueError(
                        f"Field 'acl' is required for action '{event.action.value}'"
                    )

            # 3. Симуляція обробки та індексації у Vector DB
            chunks_processed = 0
            if event.action in [ActionEnum.CREATE, ActionEnum.UPDATE]:
                chunks_processed = 1  # спрощено для mock

            result = IngestionResult(
                run_id=current_run_id,
                document_id=event.document_id,
                status="SUCCESS",
                source_system=event.source_system.value,
                action=event.action.value,
                chunks_count=chunks_processed,
                timestamp=timestamp_now,
            )
            self._log_outcome(result)
            return result

        except Exception as exc:
            error_msg = str(exc)
            result = IngestionResult(
                run_id=current_run_id,
                document_id=doc_id,
                status="FAILURE",
                source_system=raw_event.get("source_system"),
                action=raw_event.get("action"),
                error_message=error_msg,
                timestamp=timestamp_now,
            )

            # Логування невдалого запуску
            self._log_outcome(result)

            # Фіксація невдалого документа для можливості повторної обробки (Retry/DLQ)
            self.retry_store.add_failed_document(
                document_id=doc_id,
                run_id=current_run_id,
                error_reason=error_msg,
                raw_event=raw_event,
            )

            return result