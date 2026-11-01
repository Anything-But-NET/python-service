import uuid
import time
from typing import Annotated, Literal, Optional
from enum import Enum
from pydantic import BaseModel, Field, TypeAdapter


class AclPayload(BaseModel):
    allowed_groups: list[str] = Field(default_factory=list)


class DocumentUpsertEvent(BaseModel):
    sync_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    action: Literal["create", "update"]

    source_system: str = "unknown"
    title: str = "Untitled"
    url: str = ""
    raw_content: str = ""
    last_updated: str = ""

    acl: AclPayload = Field(default_factory=AclPayload)


class DocumentDeleteEvent(BaseModel):
    sync_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    action: Literal["delete"]


class DocumentUpdateAclEvent(BaseModel):
    sync_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    action: Literal["update_acl"]
    acl: AclPayload = Field(default_factory=AclPayload)


KafkaNewFileDataEvent = Annotated[
    DocumentUpsertEvent | DocumentDeleteEvent | DocumentUpdateAclEvent,
    Field(discriminator="action"),
]

event_adapter: TypeAdapter[KafkaNewFileDataEvent] = TypeAdapter(KafkaNewFileDataEvent)


#Ingest Error Handling
class SyncStatusEnum(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED_FETCH = "FAILED_FETCH"
    FAILED_PARSE = "FAILED_PARSE"
    FAILED_STORAGE = "FAILED_STORAGE"


class KafkaSyncStatusEvent(BaseModel):
    sync_id: str
    document_id: str
    status: SyncStatusEnum
    error_message: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time()))