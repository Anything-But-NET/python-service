import uuid
from typing import Annotated, Literal

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
