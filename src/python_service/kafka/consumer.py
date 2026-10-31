import asyncio
import json
import logging
import time

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

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
from python_service.services.transform_service import VectorizationService

logger = logging.getLogger(__name__)


class IngestionKafkaWorker:
    def __init__(
            self,
            parser: MarkdownParserService,
            vectorizer: VectorizationService,
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
        self.sync_status_topic = "rag.sync.status.v1"
        self._stop_event = asyncio.Event()

        self.consumer = Consumer({
            "bootstrap.servers": kafka_brokers or settings.kafka_brokers,
            "group.id": group_id or settings.kafka_group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        })

        self.producer = Producer({
            "bootstrap.servers": kafka_brokers or settings.kafka_brokers,
        })

    def _send_sync_status(self, sync_id: str, document_id: str, status: str, error_message: str = None) -> None:
        """Helper method to push status back to Java (rag.sync.status.v1)"""
        payload = {
            "sync_id": sync_id,
            "document_id": document_id,
            "status": status,
            "error_message": error_message,
            "timestamp": int(time.time())
        }
        try:
            self.producer.produce(
                topic=self.sync_status_topic,
                value=json.dumps(payload).encode("utf-8")
            )
            self.producer.poll(0)
            logger.debug(f"[{sync_id}] Sent sync status {status} for {document_id}")
        except Exception as e:
            logger.error(f"[{sync_id}] Failed to send sync status: {e}")

    async def _handle_event(self, event: KafkaNewFileDataEvent) -> None:
        if isinstance(event, DocumentUpsertEvent):
            logger.info(f"[{event.sync_id}] UPSERT for document_id: {event.document_id}")

            try:
                # PHASE 1: Parse and Vectorize (CPU-heavy)
                try:
                    ast = await asyncio.to_thread(self.parser.parse, event.raw_content)
                    vec_ast = await asyncio.to_thread(self.vectorizer.vectorize_tree, ast)
                except Exception as e:
                    self._send_sync_status(event.sync_id, event.document_id, "FAILED_PARSE", str(e))
                    raise

                # PHASE 2: Storage (DB heavy)
                try:
                    await self.db.save_document_tree(node=vec_ast, document_id=event.document_id)
                except Exception as e:
                    self._send_sync_status(event.sync_id, event.document_id, "FAILED_STORAGE", str(e))
                    raise

                # SUCCESS
                self._send_sync_status(event.sync_id, event.document_id, "COMPLETED")
                logger.info(f"[{event.sync_id}] Successfully saved: {event.document_id}")

            except Exception:
                # Catching re-raised exceptions just to let the caller know it failed
                raise

        elif isinstance(event, DocumentDeleteEvent):
            logger.info(f"[{event.sync_id}] DELETE for document_id: {event.document_id}")
            try:
                await self.db.delete_document(event.document_id)
                self._send_sync_status(event.sync_id, event.document_id, "COMPLETED")
                logger.info(f"[{event.sync_id}] Successfully deleted: {event.document_id}")
            except Exception as e:
                self._send_sync_status(event.sync_id, event.document_id, "FAILED_STORAGE", str(e))
                raise

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

                except Exception as e:
                    headers = msg.headers() or []
                    headers.extend([
                        ('error_reason', str(e).encode('utf-8')),
                        ('original_topic', msg.topic().encode('utf-8')),
                        ('original_partition', str(msg.partition()).encode('utf-8')),
                        ('original_offset', str(msg.offset()).encode('utf-8'))
                    ])

                    self.producer.produce(
                        topic=self.topic + ".dlt",
                        key=msg.key(),
                        value=msg.value(),
                        headers=headers
                    )
                    self.producer.poll(0)
                    logger.error(f"Failed processing message: {e}", exc_info=True)
                finally:
                    await asyncio.to_thread(self.consumer.commit, asynchronous=False)
        except KafkaException as e:
            logger.error(f"Kafka exception waiting: {e}", exc_info=True)
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Kafka consumer task cancelled, shutting down...")
        finally:
            await asyncio.to_thread(self.consumer.close)

    def stop(self) -> None:
        self._stop_event.set()