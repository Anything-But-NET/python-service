import uuid
from collections import defaultdict, deque
from collections.abc import Iterable

import numpy as np
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import selectinload

from python_service.config import get_settings
from python_service.db.entities import (
    Base,
    Chunks,
    VectorNode,
    vector_chunk_association,
)
from python_service.models.domain import VectorizedSegmentNode

SHARP_CHILD_SCORE_DIFF = 0.06
DEFAULT_SCORE_DECAY = 0.001


def _to_uuid(val: uuid.UUID | str) -> uuid.UUID:
    return val if isinstance(val, uuid.UUID) else uuid.UUID(str(val))


def _to_vector_list(vector: np.ndarray | list[float]) -> list[float]:
    return vector.tolist() if isinstance(vector, np.ndarray) else [float(x) for x in vector]


def _normalize_db_url(db_url: str) -> str:
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if db_url.startswith("postgresql+psycopg2://"):
        return db_url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
    return db_url


class RagDbService:
    def __init__(self, db_url: str | AsyncEngine | None = None) -> None:
        resolved_url = db_url if db_url is not None else get_settings().db_url
        if isinstance(resolved_url, str):
            self.engine: AsyncEngine = create_async_engine(_normalize_db_url(resolved_url))
        else:
            self.engine = resolved_url

        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_schema(self) -> None:
        async with self.engine.begin() as conn:
            if self.engine.dialect.name == "postgresql":
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def _clear_document_data(self, session: AsyncSession, document_id: str) -> None:
        chunk_ids_subq = select(Chunks.id).where(Chunks.document_id == document_id).scalar_subquery()

        node_ids_subq = select(vector_chunk_association.c.vector_node_id).where(
            vector_chunk_association.c.chunk_id.in_(chunk_ids_subq)
        ).scalar_subquery()

        await session.execute(delete(VectorNode).where(VectorNode.id.in_(node_ids_subq)))
        await session.execute(delete(Chunks).where(Chunks.document_id == document_id))

    async def _save_leaf_chunks(
        self,
        session: AsyncSession,
        valid_content: list[tuple[str, np.ndarray]],
        document_id: str,
        parent_id: uuid.UUID | None,
    ) -> list[uuid.UUID]:
        chunk_ids: list[uuid.UUID] = []
        for chunk_text, chunk_vec in valid_content:
            chunk_id = uuid.uuid4()
            chunk = Chunks(id=chunk_id, text=chunk_text, document_id=document_id)
            leaf_node = VectorNode(
                id=chunk_id,
                parent_id=parent_id,
                children_count=0,
                embedding=_to_vector_list(chunk_vec),
            )
            session.add(chunk)
            session.add(leaf_node)
            await session.flush()

            await session.execute(
                vector_chunk_association.insert().values(
                    vector_node_id=chunk_id,
                    chunk_id=chunk_id,
                    position=0,
                )
            )
            chunk_ids.append(chunk_id)
        return chunk_ids

    async def _save_node_recursive(
        self,
        session: AsyncSession,
        node: VectorizedSegmentNode,
        document_id: str,
        parent_id: uuid.UUID | None = None,
    ) -> list[uuid.UUID]:
        valid_content = [(text_str, vec) for text_str, vec in node.content if text_str.strip()]
        valid_children = [child for child in node.children if child.embedding is not None]

        if getattr(node, "level", None) == 0:
            all_chunk_ids = await self._save_leaf_chunks(session, valid_content, document_id, None)
            for child in valid_children:
                all_chunk_ids.extend(
                    await self._save_node_recursive(session, child, document_id, None)
                )
            return all_chunk_ids

        if node.embedding is None:
            return []

        node_id = uuid.uuid4()
        direct_children_count = len(valid_content) + len(valid_children)
        all_chunk_ids: list[uuid.UUID] = []

        parent_node = VectorNode(
            id=node_id,
            parent_id=parent_id,
            children_count=direct_children_count,
            embedding=_to_vector_list(node.embedding),
        )
        session.add(parent_node)
        await session.flush()

        if not valid_content and getattr(node, "level", 0) > 0 and getattr(node, "name", "").strip():
            header_chunk_id = uuid.uuid4()
            header_chunk = Chunks(
                id=header_chunk_id,
                text=f"{'#' * node.level} {node.name.strip()}",
                document_id=document_id,
            )
            session.add(header_chunk)
            await session.flush()
            all_chunk_ids.append(header_chunk_id)

        all_chunk_ids.extend(
            await self._save_leaf_chunks(session, valid_content, document_id, node_id)
        )

        for child in valid_children:
            all_chunk_ids.extend(
                await self._save_node_recursive(session, child, document_id, node_id)
            )

        if all_chunk_ids:
            await session.execute(
                vector_chunk_association.insert(),
                [
                    {"vector_node_id": node_id, "chunk_id": cid, "position": pos}
                    for pos, cid in enumerate(all_chunk_ids)
                ],
            )
        else:
            await session.delete(parent_node)

        return all_chunk_ids

    async def save_document_tree(
        self,
        node: VectorizedSegmentNode,
        document_id: str,
    ) -> list[uuid.UUID]:
        async with self.session_factory() as session:
            try:
                await self._clear_document_data(session, document_id)

                chunk_ids = await self._save_node_recursive(
                    session=session,
                    node=node,
                    document_id=document_id,
                    parent_id=None,
                )
                await session.commit()
                return chunk_ids
            except Exception:
                await session.rollback()
                raise

    async def get_point(self, point_id: uuid.UUID | str) -> VectorNode | None:
        pid = _to_uuid(point_id)
        async with self.session_factory() as session:
            stmt = (
                select(VectorNode)
                .options(selectinload(VectorNode.chunks))
                .where(VectorNode.id == pid)
            )
            result = await session.scalars(stmt)
            return result.first()

    async def search_points(
        self,
        vector: np.ndarray | list[float],
        limit: int = 20,
        only_leaves: bool = True,
    ) -> list[tuple[VectorNode, float]]:
        vec_list = _to_vector_list(vector)
        distance_expr = VectorNode.embedding.cosine_distance(vec_list)
        score_expr = (1.0 - distance_expr).label("score")

        stmt = (
            select(VectorNode, score_expr)
            .options(selectinload(VectorNode.chunks))
            .order_by(distance_expr)
            .limit(limit)
        )
        if only_leaves:
            stmt = stmt.where(VectorNode.children_count == 0)

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            rows = result.all()
            return [(row[0], float(row[1])) for row in rows]

    async def auto_merge_points(
        self,
        points: Iterable[tuple[VectorNode, float] | VectorNode],
        query_vector: np.ndarray | list[float],
        threshold: float = 0.5,
    ) -> list[tuple[VectorNode, float]]:
        q_vec = _to_vector_list(query_vector)

        current_points: dict[uuid.UUID, VectorNode] = {}
        nodes_cache: dict[uuid.UUID, VectorNode] = {}
        active_children: dict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
        queue: deque[uuid.UUID] = deque()
        in_queue: set[uuid.UUID] = set()
        scores: dict[uuid.UUID, float] = {}
        leaf_scores: dict[uuid.UUID, float] = {}

        for idx, item in enumerate(points):
            if isinstance(item, tuple):
                point, initial_score = item[0], float(item[1])
            else:
                point = item
                initial_score = float(1.0 - idx * DEFAULT_SCORE_DECAY)

            pid = point.id
            current_points[pid] = point
            nodes_cache[pid] = point
            scores[pid] = initial_score
            leaf_scores[pid] = initial_score

            if point.parent_id is not None:
                active_children[point.parent_id].add(pid)
                if point.parent_id not in in_queue:
                    queue.append(point.parent_id)
                    in_queue.add(point.parent_id)

        global_best_score = max(scores.values()) if scores else 0.0

        while queue:
            parent_id = queue.popleft()
            in_queue.discard(parent_id)

            children_ids = active_children.get(parent_id)
            if not children_ids:
                continue

            parent_point = nodes_cache.get(parent_id) or await self.get_point(parent_id)
            if parent_point is None:
                continue
            nodes_cache[parent_id] = parent_point

            total_children = parent_point.children_count
            if not total_children:
                continue

            best_child_score = max(scores.get(cid, 0.0) for cid in children_ids)
            best_raw_leaf_score = max(leaf_scores.get(cid, 0.0) for cid in children_ids)

            parent_score = float(np.dot(q_vec, parent_point.embedding))

            has_quorum = (len(children_ids) / total_children) > threshold
            is_competitive_with_top = (global_best_score - parent_score) <= SHARP_CHILD_SCORE_DIFF
            parent_is_better = (parent_score > best_child_score) and is_competitive_with_top
            child_is_much_sharper = (best_child_score - parent_score) > SHARP_CHILD_SCORE_DIFF

            if (has_quorum and not child_is_much_sharper) or parent_is_better:
                for cid in children_ids:
                    current_points.pop(cid, None)
                    scores.pop(cid, None)
                    leaf_scores.pop(cid, None)

                del active_children[parent_id]

                current_points[parent_id] = parent_point
                scores[parent_id] = max(best_child_score, parent_score)
                leaf_scores[parent_id] = best_raw_leaf_score

                grandpa_id = parent_point.parent_id
                if grandpa_id is not None:
                    active_children[grandpa_id].add(parent_id)
                    if grandpa_id not in in_queue:
                        queue.append(grandpa_id)
                        in_queue.add(grandpa_id)

        sorted_by_score = sorted(
            current_points.values(),
            key=lambda p: scores.get(p.id, 0.0),
            reverse=True,
        )
        final_ids = []
        covered_chunks: set[uuid.UUID] = set()

        for p in sorted_by_score:
            p_chunk_ids = {c.id for c in p.chunks}
            if p_chunk_ids and not p_chunk_ids.isdisjoint(covered_chunks):
                continue
            final_ids.append(p.id)
            covered_chunks.update(p_chunk_ids)

        return [(current_points[pid], scores[pid]) for pid in final_ids]

    async def query_with_auto_merge(
        self,
        vector: np.ndarray | list[float],
        limit: int = 20,
        threshold: float = 0.5,
    ) -> list[dict]:
        initial_points = await self.search_points(vector, limit=limit, only_leaves=True)
        merged_points = await self.auto_merge_points(
            initial_points,
            query_vector=vector,
            threshold=threshold,
        )

        results: list[dict] = []
        for point, score in merged_points:
            chunk_ids = [str(c.id) for c in point.chunks]
            chunk_texts = [c.text for c in point.chunks]

            document_ids = list(dict.fromkeys(c.document_id for c in point.chunks))

            results.append({
                "point_id": str(point.id),
                "parent_id": str(point.parent_id) if point.parent_id else None,
                "score": score,
                "children_count": point.children_count,
                "chunk_ids": chunk_ids,
                "chunks": chunk_texts,
                "document_id": document_ids[0] if document_ids else None,
                "text": "\n".join(chunk_texts),
            })

        return results

    async def delete_document(self, document_id: str) -> None:
        async with self.session_factory() as session:
            try:
                await self._clear_document_data(session, document_id)
                await session.commit()
            except Exception:
                await session.rollback()
                raise
