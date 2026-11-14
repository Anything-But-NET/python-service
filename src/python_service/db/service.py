import logging
import uuid
from collections import defaultdict, deque
from collections.abc import Iterable
from datetime import UTC, datetime

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
    RagDocument,
    VectorNode,
    vector_chunk_association,
)
from python_service.models.domain import VectorizedSegmentNode
import json
from sqlalchemy.dialects.postgresql import insert
from python_service.db.entities import PromptVersionEntity
from python_service.models.domain import PromptVersion

SHARP_CHILD_SCORE_DIFF = 0.06
DEFAULT_SCORE_DECAY = 0.001

logger = logging.getLogger(__name__)


def _normalize_groups(groups: Iterable[str]) -> list[str]:
    """Strip, drop blanks and de-duplicate while preserving order."""
    return list(dict.fromkeys(
        group.strip()
        for group in groups
        if group and group.strip()
    ))


def _to_aware_datetime(value: datetime | str | None) -> datetime | None:
    """Parse ISO strings and treat naive datetimes as UTC."""
    if isinstance(value, str):
        value = value.strip()
        value = datetime.fromisoformat(value) if value else None
    if value is not None and value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value


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

def _scrub_metadata(metadata: dict | list | None) -> dict | list:
    if not metadata:
        return metadata if metadata is not None else {}

    forbidden_substrings = ["api_key", "secret", "api_token", "auth_token", "password", "credentials"]

    if isinstance(metadata, dict):
        for key, value in metadata.items():
            key_lower = str(key).lower()
            if any(sub in key_lower for sub in forbidden_substrings):
                raise ValueError(f"Detected potential secret in metadata key: '{key}'. Secrets are strictly prohibited.")

            if isinstance(value, (dict, list)):
                _scrub_metadata(value)
            else:
                val_str = str(value)
                if val_str.startswith("sk-") and len(val_str) > 40:
                    raise ValueError(f"Detected potential API token in metadata value for key: '{key}'.")
    elif isinstance(metadata, list):
        for item in metadata:
            if isinstance(item, (dict, list)):
                _scrub_metadata(item)

    return metadata

def _generate_prompt_id(prompt_text: str, metadata: dict) -> uuid.UUID:
    #Generates a deterministic UUIDv5 based on text and metadata
    meta_str = json.dumps(metadata, sort_keys=True)
    content = f"{prompt_text}|{meta_str}"

    NAMESPACE_PROMPT = uuid.uuid5(uuid.NAMESPACE_OID, "rag-prompt-versioning")
    return uuid.uuid5(NAMESPACE_PROMPT, content)

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

    async def update_document_acl(
        self,
        document_id: str,
        allowed_groups: list[str],
    ) -> bool:
        """Replace the ACL of an existing document.

        An empty ACL is stored as-is (default deny: nobody can retrieve it).
        Returns False when the document is unknown; its next upsert carries
        the full ACL anyway, so this is not treated as a storage failure.
        """
        groups = _normalize_groups(allowed_groups)

        async with self.session_factory() as session:
            try:
                document = await session.get(
                    RagDocument,
                    document_id,
                    with_for_update=True,
                )

                if document is None:
                    logger.warning(
                        "UPDATE_ACL skipped: document %s is not indexed yet",
                        document_id,
                    )
                    return False

                document.allowed_groups = groups
                document.last_sync_at = datetime.now(UTC)

                await session.commit()
                return True

            except Exception:
                await session.rollback()
                raise

    async def init_schema(self) -> None:
        # Done by Alembic migrations.
        pass

    async def close(self) -> None:
        await self.engine.dispose()

    async def _clear_document_data(
        self,
        session: AsyncSession,
        document_id: str,
    ) -> None:
        await session.execute(
            delete(VectorNode).where(VectorNode.document_id == document_id)
        )
        await session.execute(
            delete(Chunks).where(Chunks.document_id == document_id)
        )

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
                document_id=document_id,
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
            document_id=document_id,
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

    async def _upsert_rag_document(
        self,
        session: AsyncSession,
        document_id: str,
        allowed_groups: list[str],
        last_updated: datetime | str | None = None,
    ) -> bool:
        """Insert or update the document row.

        Returns False (and changes nothing) when the incoming event is older
        than what is already stored, so out-of-order events cannot overwrite
        newer content or ACL. An empty ACL is stored as default deny.
        """
        groups = _normalize_groups(allowed_groups)
        last_updated = _to_aware_datetime(last_updated)

        document = await session.get(
            RagDocument,
            document_id,
            with_for_update=True,
        )

        if (
            document is not None
            and document.last_updated is not None
            and last_updated is not None
            and last_updated < _to_aware_datetime(document.last_updated)
        ):
            logger.warning(
                "Skipping stale upsert for %s (event last_updated=%s, stored=%s)",
                document_id,
                last_updated.isoformat(),
                document.last_updated.isoformat(),
            )
            return False

        if document is None:
            document = RagDocument(
                document_id=document_id,
                allowed_groups=groups,
                last_updated=last_updated,
                last_sync_at=datetime.now(UTC),
            )
            session.add(document)
        else:
            document.allowed_groups = groups
            if last_updated is not None:
                document.last_updated = last_updated
            document.last_sync_at = datetime.now(UTC)

        await session.flush()
        return True

    async def save_document_tree(
        self,
        node: VectorizedSegmentNode,
        document_id: str,
        allowed_groups: list[str],
        last_updated: datetime | str | None = None,
    ) -> list[uuid.UUID] | None:
        """Persist the document tree. Returns None if the event was stale."""
        async with self.session_factory() as session:
            try:
                applied = await self._upsert_rag_document(
                    session=session,
                    document_id=document_id,
                    allowed_groups=allowed_groups,
                    last_updated=last_updated,
                )
                if not applied:
                    await session.rollback()
                    return None

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

    async def get_point(
        self,
        point_id: uuid.UUID | str,
        allowed_groups: list[str],
    ) -> VectorNode | None:
        groups = _normalize_groups(allowed_groups)

        if not groups:
            return None

        pid = _to_uuid(point_id)

        async with self.session_factory() as session:
            stmt = (
                select(VectorNode)
                .join(
                    RagDocument,
                    RagDocument.document_id == VectorNode.document_id,
                )
                .options(selectinload(VectorNode.chunks))
                .where(
                    VectorNode.id == pid,
                    RagDocument.allowed_groups.overlap(groups),
                )
            )
            result = await session.scalars(stmt)
            return result.first()

    async def search_points(
        self,
        vector: np.ndarray | list[float],
        allowed_groups: list[str],
        limit: int = 20,
        only_leaves: bool = True,
    ) -> list[tuple[VectorNode, float]]:
        groups = _normalize_groups(allowed_groups)

        if not groups:
            raise ValueError("Cannot search without user permission groups")

        vec_list = _to_vector_list(vector)
        distance_expr = VectorNode.embedding.cosine_distance(vec_list)
        score_expr = (1.0 - distance_expr).label("score")

        stmt = (
            select(VectorNode, score_expr)
            .join(
                RagDocument,
                RagDocument.document_id == VectorNode.document_id,
            )
            .where(RagDocument.allowed_groups.overlap(groups))
            .options(selectinload(VectorNode.chunks))
            .order_by(distance_expr)
            .limit(limit)
        )

        if only_leaves:
            stmt = stmt.where(VectorNode.children_count == 0)

        async with self.session_factory() as session:
            if self.engine.dialect.name == "postgresql":
                # The ACL filter is applied after the HNSW scan. Without an
                # iterative scan a restrictive ACL can leave fewer than `limit`
                # (or zero) rows even when accessible matches exist.
                # Requires pgvector >= 0.8. SET LOCAL is scoped to this transaction.
                await session.execute(
                    text("SET LOCAL hnsw.iterative_scan = relaxed_order")
                )
            result = await session.execute(stmt)
            rows = result.all()
            # relaxed_order may return rows slightly out of order; re-sort.
            return sorted(
                ((row[0], float(row[1])) for row in rows),
                key=lambda item: item[1],
                reverse=True,
            )

    async def auto_merge_points(
        self,
        points: Iterable[tuple[VectorNode, float] | VectorNode],
        query_vector: np.ndarray | list[float],
        allowed_groups: list[str],
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

            parent_point = nodes_cache.get(parent_id) or await self.get_point(
                parent_id,
                allowed_groups,
            )
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
        allowed_groups: list[str],
        limit: int = 20,
        threshold: float = 0.5,
    ) -> list[dict]:
        initial_points = await self.search_points(
            vector,
            allowed_groups=allowed_groups,
            limit=limit,
            only_leaves=True,
        )
        merged_points = await self.auto_merge_points(
            initial_points,
            query_vector=vector,
            allowed_groups=allowed_groups,
            threshold=threshold,
        )

        results: list[dict] = []
        for point, score in merged_points:
            # Every vector node belongs to exactly one document (FK on
            # vector_nodes.document_id), so provenance is unambiguous.
            chunk_texts = [chunk.text for chunk in point.chunks]

            results.append({
                "point_id": str(point.id),
                "parent_id": str(point.parent_id) if point.parent_id else None,
                "score": score,
                "children_count": point.children_count,
                "chunk_ids": [str(chunk.id) for chunk in point.chunks],
                "chunks": chunk_texts,
                "document_id": point.document_id,
                "text": "\n".join(chunk_texts),
            })

        return results

    async def delete_document(self, document_id: str) -> None:
        async with self.session_factory() as session:
            try:
                await self._clear_document_data(session, document_id)
                document = await session.get(RagDocument, document_id)
                if document is not None:
                    await session.delete(document)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_or_create_prompt_version(
            self,
            prompt_text: str,
            metadata: dict | None = None,
    ) -> PromptVersion:
        """
        Register new prompt version or return existing ID.
        Uses Content-Addressable Storage pattern (UUIDv5).
        """
        clean_meta = _scrub_metadata(metadata)

        prompt_id = _generate_prompt_id(prompt_text, clean_meta)

        async with self.session_factory() as session:
            try:
                # save (PostgreSQL INSERT ON CONFLICT DO NOTHING)
                stmt = insert(PromptVersionEntity).values(
                    id=prompt_id,
                    prompt_text=prompt_text,
                    prompt_metadata=clean_meta,
                ).on_conflict_do_nothing(index_elements=['id'])

                await session.execute(stmt)
                await session.commit()

                return PromptVersion(
                    id=prompt_id,
                    prompt_text=prompt_text,
                    metadata=clean_meta,
                )
            except Exception:
                await session.rollback()
                raise
