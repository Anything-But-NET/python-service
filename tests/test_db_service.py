import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from python_service.db.entities import Chunks, VectorNode
from python_service.db.service import (
    RagDbService,
    _normalize_db_url,
    _to_uuid,
    _to_vector_list,
)
from python_service.models.domain import VectorizedSegmentNode


def test_helpers():
    # _normalize_db_url
    assert _normalize_db_url("postgresql://user:pass@host/db") == "postgresql+psycopg://user:pass@host/db"
    assert _normalize_db_url("postgresql+psycopg2://user:pass@host/db") == "postgresql+psycopg://user:pass@host/db"
    assert _normalize_db_url("postgresql+psycopg://user:pass@host/db") == "postgresql+psycopg://user:pass@host/db"

    # _to_uuid
    uid = uuid.uuid4()
    assert _to_uuid(uid) == uid
    assert _to_uuid(str(uid)) == uid

    # _to_vector_list
    arr = np.array([1.0, 2.0, 3.0])
    assert _to_vector_list(arr) == [1.0, 2.0, 3.0]
    assert _to_vector_list([1.0, 2.0]) == [1.0, 2.0]


@pytest.fixture
def mock_db_service():
    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"
    mock_engine.dispose = AsyncMock()

    service = RagDbService(db_url=mock_engine)

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    # async with service.session_factory() as session
    service.session_factory = MagicMock()
    service.session_factory.return_value.__aenter__.return_value = mock_session
    service.session_factory.return_value.__aexit__.return_value = None

    return service, mock_engine, mock_session


@pytest.mark.asyncio
async def test_init_schema_and_close(mock_db_service):
    service, mock_engine, _ = mock_db_service

    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__.return_value = mock_conn

    # PostgreSQL dialect
    await service.init_schema()
    mock_conn.execute.assert_awaited_once()
    mock_conn.run_sync.assert_awaited_once()

    # Non-postgres dialect
    mock_engine.dialect.name = "sqlite"
    mock_conn.reset_mock()
    await service.init_schema()
    mock_conn.execute.assert_not_awaited()
    mock_conn.run_sync.assert_awaited_once()

    await service.close()
    mock_engine.dispose.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_document_tree_success_and_rollback(mock_db_service):
    service, _, mock_session = mock_db_service

    # Create a tree: Root (level 0) -> Section 1 (level 1) with chunk content
    child_vec_node = VectorizedSegmentNode(
        name="Section 1",
        level=1,
        embedding=np.zeros(1024),
        content=[("Chunk 1 text", np.zeros(1024))],
        children=[],
    )
    root = VectorizedSegmentNode(
        name="Root",
        level=0,
        embedding=np.zeros(1024),
        content=[("Root preamble chunk", np.zeros(1024))],
        children=[child_vec_node],
    )

    chunk_ids = await service.save_document_tree(root, document_id="doc-1")
    assert len(chunk_ids) >= 2
    mock_session.commit.assert_awaited_once()

    # Test rollback on error
    mock_session.commit.side_effect = RuntimeError("DB commit failed")
    with pytest.raises(RuntimeError):
        await service.save_document_tree(root, document_id="doc-1")
    mock_session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_node_recursive_empty_content_with_header(mock_db_service):
    service, _, mock_session = mock_db_service

    # Section with empty content but with name/level > 0 (generates header chunk)
    header_only_node = VectorizedSegmentNode(
        name="Header Only",
        level=2,
        embedding=np.zeros(1024),
        content=[],
        children=[],
    )

    ids = await service._save_node_recursive(
        session=mock_session,
        node=header_only_node,
        document_id="doc-2",
    )
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_save_node_recursive_none_embedding(mock_db_service):
    service, _, mock_session = mock_db_service

    no_emb_node = VectorizedSegmentNode(
        name="No Embedding",
        level=1,
        embedding=None,
        content=[],
        children=[],
    )
    ids = await service._save_node_recursive(
        session=mock_session,
        node=no_emb_node,
        document_id="doc-3",
    )
    assert ids == []


@pytest.mark.asyncio
async def test_delete_document(mock_db_service):
    service, _, mock_session = mock_db_service

    await service.delete_document("doc-del")
    mock_session.execute.assert_awaited()
    mock_session.commit.assert_awaited_once()

    # Test error and rollback
    mock_session.commit.side_effect = RuntimeError("Delete error")
    with pytest.raises(RuntimeError):
        await service.delete_document("doc-del")
    mock_session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_point(mock_db_service):
    service, _, mock_session = mock_db_service

    fake_node = VectorNode(id=uuid.uuid4(), children_count=0, embedding=[0.0] * 1024)
    scalars_mock = MagicMock()
    scalars_mock.first.return_value = fake_node
    mock_session.scalars.return_value = scalars_mock

    node = await service.get_point(fake_node.id)
    assert node is fake_node


@pytest.mark.asyncio
async def test_search_points(mock_db_service):
    service, _, mock_session = mock_db_service

    fake_node = VectorNode(id=uuid.uuid4(), children_count=0, embedding=[0.0] * 1024)
    execute_mock = MagicMock()
    execute_mock.all.return_value = [(fake_node, 0.88)]
    mock_session.execute.return_value = execute_mock

    # Test with only_leaves=True and False
    res_leaves = await service.search_points(np.zeros(1024), limit=5, only_leaves=True)
    assert len(res_leaves) == 1
    assert res_leaves[0][1] == 0.88

    res_all = await service.search_points(np.zeros(1024), limit=5, only_leaves=False)
    assert len(res_all) == 1


@pytest.mark.asyncio
async def test_auto_merge_points_and_query_with_auto_merge(mock_db_service):
    service, _, mock_session = mock_db_service

    # Setup parent and 2 children
    parent_id = uuid.uuid4()
    child1_id = uuid.uuid4()
    child2_id = uuid.uuid4()

    chunk1 = Chunks(id=child1_id, text="Child 1 text", document_id="doc-1")
    chunk2 = Chunks(id=child2_id, text="Child 2 text", document_id="doc-1")

    # Parent embedding unit vector along dim 0
    parent_emb = [1.0] + [0.0] * 1023
    parent_node = VectorNode(
        id=parent_id,
        parent_id=None,
        children_count=2,
        embedding=parent_emb,
        chunks=[chunk1, chunk2],
    )

    child1_node = VectorNode(
        id=child1_id,
        parent_id=parent_id,
        children_count=0,
        embedding=[0.9] + [0.0] * 1023,
        chunks=[chunk1],
    )
    child2_node = VectorNode(
        id=child2_id,
        parent_id=parent_id,
        children_count=0,
        embedding=[0.85] + [0.0] * 1023,
        chunks=[chunk2],
    )

    # Mock get_point when parent is resolved from DB
    async def fake_get_point(pid):
        if pid == parent_id:
            return parent_node
        return None

    service.get_point = AsyncMock(side_effect=fake_get_point)

    # Initial points: both child1 and child2 are retrieved
    # Query vector is [1.0, 0.0, ...] -> parent_score will be 1.0 (high score)
    query_vector = np.array([1.0] + [0.0] * 1023)
    initial_points = [(child1_node, 0.8), (child2_node, 0.75)]

    merged = await service.auto_merge_points(
        points=initial_points,
        query_vector=query_vector,
        threshold=0.5,
    )
    # Both children should be auto-merged into parent_node
    assert len(merged) == 1
    winner_node, score = merged[0]
    assert winner_node.id == parent_id

    # Test query_with_auto_merge
    with patch.object(service, "search_points", return_value=initial_points):
        dict_results = await service.query_with_auto_merge(query_vector, limit=10, threshold=0.5)
        assert len(dict_results) == 1
        res = dict_results[0]
        assert res["point_id"] == str(parent_id)
        assert res["document_id"] == "doc-1"
        assert "Child 1 text\nChild 2 text" in res["text"]
