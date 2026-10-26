import importlib
import json
import sys
import unittest
from unittest.mock import MagicMock, patch


class KafkaSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patches = [
            patch("langchain_openai.OpenAIEmbeddings"),
            patch("langchain_postgres.vectorstores.PGVector"),
            patch("sqlalchemy.create_engine"),
        ]
        for p in cls.patches:
            p.start()

        sys.modules.pop("python_service.kafka_sync", None)
        cls.sync = importlib.import_module("python_service.kafka_sync")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("python_service.kafka_sync", None)
        for p in reversed(cls.patches):
            p.stop()

    def setUp(self):
        self.conn = MagicMock()
        self.sync.engine = MagicMock()
        self.sync.engine.begin.return_value.__enter__.return_value = self.conn
        self.sync.vector_store = MagicMock()

    def message(self, payload, key=None):
        msg = MagicMock()
        msg.value = (
            json.dumps(payload).encode("utf-8")
            if payload is not None else None
        )
        msg.key = key
        return msg

    def test_update_replaces_existing_document_without_duplicate_chunks(self):
        payload = {
            "document_id": "doc-1",
            "raw_content": "Updated document content",
            "acl": {"allowed_groups": ["HR", "IT"]},
            "is_deleted": False,
        }

        self.sync.process_message(self.message(payload))

        self.conn.execute.assert_called_once()
        self.sync.vector_store.add_documents.assert_called_once()

        kwargs = self.sync.vector_store.add_documents.call_args.kwargs
        self.assertEqual(kwargs["ids"], ["doc-1_chunk_0"])
        self.assertEqual(
            kwargs["documents"][0].metadata,
            {"doc_id": "doc-1", "allowed_roles": ["HR", "IT"]},
        )
        self.assertEqual(
            kwargs["documents"][0].page_content,
            "Updated document content",
        )

    def test_reprocessing_uses_same_chunk_ids(self):
        payload = {
            "document_id": "doc-2",
            "raw_content": "Same content",
            "acl": {"allowed_groups": ["Owners"]},
        }

        self.sync.process_message(self.message(payload))
        first_ids = self.sync.vector_store.add_documents.call_args.kwargs["ids"]

        self.sync.process_message(self.message(payload))
        second_ids = self.sync.vector_store.add_documents.call_args.kwargs["ids"]

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(self.conn.execute.call_count, 2)
        self.assertEqual(self.sync.vector_store.add_documents.call_count, 2)

    def test_deleted_document_is_removed_from_vector_index(self):
        self.sync.process_message(self.message({
            "document_id": "doc-3",
            "raw_content": "",
            "is_deleted": True,
        }))

        self.conn.execute.assert_called_once()
        self.assertIn(
            "DELETE FROM langchain_pg_embedding",
            str(self.conn.execute.call_args.args[0]),
        )
        self.assertEqual(
            self.conn.execute.call_args.args[1],
            {"doc_id": "doc-3"},
        )
        self.sync.vector_store.add_documents.assert_not_called()

    def test_kafka_tombstone_removes_document_by_key(self):
        self.sync.process_message(
            self.message(None, key=b'"doc-4"')
        )

        self.conn.execute.assert_called_once()
        self.assertEqual(
            self.conn.execute.call_args.args[1],
            {"doc_id": "doc-4"},
        )
        self.sync.vector_store.add_documents.assert_not_called()

    def test_empty_content_does_not_create_vectors(self):
        self.sync.process_message(self.message({
            "document_id": "doc-5",
            "raw_content": "",
            "acl": {"allowed_groups": ["IT"]},
        }))

        self.conn.execute.assert_called_once()
        self.sync.vector_store.add_documents.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
