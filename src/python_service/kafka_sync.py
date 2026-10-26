import os
import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from kafka import KafkaConsumer
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = os.getenv("VECTOR_DB_URL", "postgresql+psycopg2://postgres:example@localhost:5432/copilot_db")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")

class AclRules(BaseModel):
    allowed_groups: List[str] = Field(default_factory=list)

class SyncEvent(BaseModel):
    document_id: str
    raw_content: Optional[str] = ""
    acl: Optional[AclRules] = None

vector_store = PGVector(
    embeddings=OpenAIEmbeddings(),
    collection_name="sharepoint_docs",
    connection=DB_URL,
)
engine = create_engine(DB_URL)

def process_message(msg):
    if msg.value is None:
        if msg.key:
            doc_id = msg.key.decode('utf-8').replace('"', '')
            with engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'doc_id' = :doc_id"),
                    {"doc_id": doc_id}
                )
        return

    raw_payload = json.loads(msg.value.decode('utf-8'))
    
    if "message" in raw_payload:
        raw_payload = raw_payload["message"]

    event = SyncEvent(**raw_payload)
    doc_id = event.document_id

    if raw_payload.get("is_deleted", False):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM langchain_pg_embedding "
                    "WHERE cmetadata->>'doc_id' = :doc_id"
                ),
                {"doc_id": doc_id},
            )
        logger.info("Deleted document %s from vector index", doc_id)
        return

    roles = event.acl.allowed_groups if event.acl else []
    content = event.raw_content or ""

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'doc_id' = :doc_id"),
            {"doc_id": doc_id}
        )
    
    if not content:
        return

    chunk_size = 1000
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    docs = []
    ids = []
    for idx, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk, 
                metadata={"doc_id": doc_id, "allowed_roles": roles}
            )
        )
        ids.append(f"{doc_id}_chunk_{idx}")
    
    vector_store.add_documents(documents=docs, ids=ids)

def start_consumer():
    consumer = KafkaConsumer(
        'new-file-data',
        bootstrap_servers=[KAFKA_BROKER],
        group_id='python-vector-sync-group',
        enable_auto_commit=False
    )
    for message in consumer:
        try:
            process_message(message)
            consumer.commit()
        except Exception as e:
            logger.error(f"Failed: {e}")

if __name__ == "__main__":
    start_consumer()