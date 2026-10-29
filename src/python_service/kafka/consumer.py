import asyncio
import json
import logging

from confluent_kafka import Consumer, KafkaError, KafkaException

from python_service.config import get_settings
from python_service.db.service import RagDbService
from python_service.models.events import (
    DocumentDeleteEvent,
    DocumentUpdateAclEvent,
    DocumentUpsertEvent,
    KafkaNewFileDataEvent,
    event_adapter,
)
from python_service.services.parse_service import MarkdownParserService
from python_service.services.transform_service import MarkdownVectorizationService

logger = logging.getLogger(__name__)


class IngestionKafkaWorker:
    def __init__(
        self,
        parser: MarkdownParserService,
        vectorizer: MarkdownVectorizationService,
        db: RagDbService,
        kafka_brokers: str | None = None,
        group_id: str | None = None,
        topic: str | None = None,
    ):
        settings = get_settings()
        self.parser = parser
        self.vectorizer = vectorizer
        self.db = db
        self.topic = topic or settings.kafka_topic
        self._stop_event = asyncio.Event()

        self.consumer = Consumer({
            "bootstrap.servers": kafka_brokers or settings.kafka_brokers,
            "group.id": group_id or settings.kafka_group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        })

    async def _handle_event(self, event: KafkaNewFileDataEvent) -> None:
        if isinstance(event, DocumentUpsertEvent):
            logger.info(f"[{event.sync_id}] UPSERT for document_id: {event.document_id}")

            # Виконуємо CPU/GPU-важкі операції у фоновому потоці, щоб не блокувати event loop FastAPI
            ast = await asyncio.to_thread(self.parser.parse, event.raw_content)
            vec_ast = await asyncio.to_thread(self.vectorizer.vectorize_tree, ast)
            await self.db.save_document_tree(node=vec_ast, document_id=event.document_id)

            logger.info(f"[{event.sync_id}] Successfully saved: {event.document_id}")

        elif isinstance(event, DocumentDeleteEvent):
            logger.info(f"[{event.sync_id}] DELETE for document_id: {event.document_id}")
            await self.db.delete_document(event.document_id)
            logger.info(f"[{event.sync_id}] Successfully deleted: {event.document_id}")

        elif isinstance(event, DocumentUpdateAclEvent):
            logger.info(f"[{event.sync_id}] Ignoring UPDATE_ACL for {event.document_id}")

    async def start_consuming(self) -> None:
        self._stop_event.clear()
        await asyncio.to_thread(self.consumer.subscribe, [self.topic])
        logger.info(f"Listening to Kafka topic: {self.topic}...")

        try:
            while not self._stop_event.is_set():
                msg = await asyncio.to_thread(self.consumer.poll, 1.0)
                if msg is None:
                    continue

                err = msg.error()
                if err:
                    if err.code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(err)

                val = msg.value()
                if val is None:
                    continue

                raw_json = val.decode("utf-8")

                try:
                    payload_dict = json.loads(raw_json)
                    event = event_adapter.validate_python(payload_dict)

                    await self._handle_event(event)
                    await asyncio.to_thread(self.consumer.commit, asynchronous=False)

                except Exception as e:
                    logger.error(f"Failed processing message: {e}", exc_info=True)

        except asyncio.CancelledError:
            logger.info("Kafka consumer task cancelled, shutting down...")
        finally:
            await asyncio.to_thread(self.consumer.close)

    def stop(self) -> None:
        self._stop_event.set()
