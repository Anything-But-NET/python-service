import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from python_service.kafka.consumer import IngestionKafkaWorker
from python_service.models.domain import SegmentNode, VectorizedSegmentNode
from python_service.models.events import (
    DocumentDeleteEvent,
    DocumentUpdateAclEvent,
    DocumentUpsertEvent,
    event_adapter,
)
from python_service.services.parse_service import MarkdownParserService


class RefactoredServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.parser = MarkdownParserService()
        self.vectorizer = MagicMock()
        self.db = MagicMock()
        self.db.save_document_tree = AsyncMock(return_value=[])
        self.db.delete_document = AsyncMock(return_value=None)

    def test_markdown_parser_builds_segment_tree(self) -> None:
        md = "# Title\nIntro text\n## Subtitle\nDetails here"
        root = self.parser.parse(md)

        self.assertIsInstance(root, SegmentNode)
        self.assertEqual(root.level, 0)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.children[0].name, "Title")
        self.assertEqual(root.children[0].content, "Intro text")
        self.assertEqual(len(root.children[0].children), 1)
        self.assertEqual(root.children[0].children[0].name, "Subtitle")
        self.assertEqual(root.children[0].children[0].content, "Details here")

    @patch("python_service.kafka.consumer.Consumer")
    async def test_kafka_worker_handles_upsert_event(self, _mock_consumer_cls: MagicMock) -> None:
        fake_vec_node = VectorizedSegmentNode(
            name="Root",
            level=0,
            embedding=np.zeros(1024),
            content=[],
            children=[],
        )
        self.vectorizer.vectorize_tree.return_value = fake_vec_node

        worker = IngestionKafkaWorker(
            parser=self.parser,
            vectorizer=self.vectorizer,
            db=self.db,
        )

        raw_event = {
            "action": "create",
            "document_id": "doc-1",
            "raw_content": "# Heading\nContent",
            "acl": {"allowed_groups": ["HR", "IT"]},
        }
        event = event_adapter.validate_python(raw_event)
        self.assertIsInstance(event, DocumentUpsertEvent)

        await worker._handle_event(event)

        self.vectorizer.vectorize_tree.assert_called_once()
        self.db.save_document_tree.assert_awaited_once_with(
            node=fake_vec_node,
            document_id="doc-1",
        )

    @patch("python_service.kafka.consumer.Consumer")
    async def test_kafka_worker_handles_delete_event(self, _mock_consumer_cls: MagicMock) -> None:
        worker = IngestionKafkaWorker(
            parser=self.parser,
            vectorizer=self.vectorizer,
            db=self.db,
        )

        raw_event = {
            "action": "delete",
            "document_id": "doc-3",
        }
        event = event_adapter.validate_python(raw_event)
        self.assertIsInstance(event, DocumentDeleteEvent)

        await worker._handle_event(event)

        self.db.delete_document.assert_awaited_once_with("doc-3")
        self.db.save_document_tree.assert_not_awaited()

    @patch("python_service.kafka.consumer.Consumer")
    async def test_kafka_worker_ignores_update_acl_event(self, _mock_consumer_cls: MagicMock) -> None:
        worker = IngestionKafkaWorker(
            parser=self.parser,
            vectorizer=self.vectorizer,
            db=self.db,
        )

        raw_event = {
            "action": "update_acl",
            "document_id": "doc-4",
            "acl": {"allowed_groups": ["Admins"]},
        }
        event = event_adapter.validate_python(raw_event)
        self.assertIsInstance(event, DocumentUpdateAclEvent)

        await worker._handle_event(event)

        self.db.delete_document.assert_not_awaited()
        self.db.save_document_tree.assert_not_awaited()

    def test_app_and_config_initialization(self) -> None:
        from python_service.config import Settings
        from python_service.db.entities import Base, Chunks, VectorNode
        from python_service.main import create_app

        settings = Settings(DB_URL="postgresql+psycopg://user:pass@db:5432/rag", KAFKA_BROKERS="kafka:9092")
        self.assertEqual(settings.db_url, "postgresql+psycopg://user:pass@db:5432/rag")
        self.assertEqual(settings.kafka_brokers, "kafka:9092")
        self.assertIn("chunks", Base.metadata.tables)
        self.assertIn("vector_nodes", Base.metadata.tables)
        self.assertIsNotNone(Chunks)
        self.assertIsNotNone(VectorNode)

        app = create_app()
        paths = app.openapi()["paths"]
        self.assertIn("/health", paths)
        self.assertIn("/search", paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
