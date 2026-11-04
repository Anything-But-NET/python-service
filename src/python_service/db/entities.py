import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

VECTOR_SIZE = 1024


class Base(DeclarativeBase):
    metadata = MetaData(schema="rag")


vector_chunk_association = Table(
    "vector_chunk_association",
    Base.metadata,
    Column(
        "vector_node_id",
        Uuid,
        ForeignKey("rag.vector_nodes.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "chunk_id",
        Uuid,
        ForeignKey("rag.chunks.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("position", Integer, nullable=False, default=0),
)


class RagDocument(Base):
    __tablename__ = "document"

    document_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    allowed_groups: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("'{}'"),
    )
    last_updated: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_sync_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index(
            "idx_document_metadata_allowed_groups",
            "allowed_groups",
            postgresql_using="gin",
        ),
    )


class Chunks(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("rag.document.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    __table_args__ = (Index("chunk_doc_id", "document_id"),)


class VectorNode(Base):
    __tablename__ = "vector_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("rag.vector_nodes.id", ondelete="SET NULL"),
        nullable=True,
    )
    children_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(VECTOR_SIZE),
        nullable=False,
    )
    document_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("rag.document.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    chunks: Mapped[list["Chunks"]] = relationship(
        secondary=vector_chunk_association,
        order_by=vector_chunk_association.c.position,
    )
    parent: Mapped["VectorNode | None"] = relationship(
        "VectorNode",
        back_populates="children",
        remote_side=[id],
    )
    children: Mapped[list["VectorNode"]] = relationship(
        "VectorNode",
        back_populates="parent",
    )

    __table_args__ = (
        Index(
            "embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("vector_node_doc_id", "document_id"),
    )