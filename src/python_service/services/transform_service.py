import grpc
import numpy as np

from python_service.config import get_settings
from python_service.grpc_gen import vectorizer_pb2, vectorizer_pb2_grpc
from python_service.models.domain import SegmentNode, VectorizedSegmentNode


MAX_MESSAGE_LENGTH = 50 * 1024 * 1024  # 50 MB
GRPC_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
]


def _decode_vector(raw_bytes: bytes) -> np.ndarray | None:
    if not raw_bytes:
        return None
    return np.frombuffer(raw_bytes, dtype=np.float32).copy()


class VectorizationService:
    def __init__(
        self,
        grpc_target: str | None = None,
        channel: grpc.Channel | None = None,
    ):
        self.grpc_target = grpc_target or get_settings().vectorizer_grpc_target
        self._owns_channel = channel is None
        self.channel = channel or grpc.insecure_channel(
            self.grpc_target,
            options=GRPC_CHANNEL_OPTIONS,
        )
        self.stub = vectorizer_pb2_grpc.MarkdownVectorizerStub(self.channel)

    def close(self) -> None:
        if self._owns_channel and self.channel is not None:
            self.channel.close()

    def vectorize_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            response = self.stub.VectorizeQuery(
                vectorizer_pb2.QueryRequest(query=query)
            )
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError(exc.details() or "Invalid query") from exc
            raise

        vec = _decode_vector(response.embedding)
        if vec is None:
            raise ValueError("Query vectorization returned empty embedding")
        return vec

    @staticmethod
    def _flatten_tree(
        root: SegmentNode,
    ) -> tuple[list[SegmentNode], list[vectorizer_pb2.FlatNodeInput], list[int]]:
        original_nodes: list[SegmentNode] = []

        def collect_nodes(node: SegmentNode) -> None:
            original_nodes.append(node)
            for child in node.children:
                collect_nodes(child)

        collect_nodes(root)

        node_to_idx = {id(node): idx for idx, node in enumerate(original_nodes)}
        flat_inputs: list[vectorizer_pb2.FlatNodeInput] = []
        child_indices: list[int] = []

        for node in original_nodes:
            first_edge_idx = len(child_indices)
            edge_count = len(node.children)
            for child in node.children:
                child_indices.append(node_to_idx[id(child)])

            flat_inputs.append(
                vectorizer_pb2.FlatNodeInput(
                    name=node.name,
                    level=node.level,
                    content=node.content,
                    first_edge_idx=first_edge_idx,
                    edge_count=edge_count,
                )
            )

        return original_nodes, flat_inputs, child_indices

    def vectorize_tree(self, node: SegmentNode) -> VectorizedSegmentNode:
        original_nodes, flat_inputs, child_indices = self._flatten_tree(node)

        try:
            response = self.stub.VectorizeTree(
                vectorizer_pb2.TreeRequest(
                    nodes=flat_inputs,
                    child_indices=child_indices,
                )
            )
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError(exc.details() or "Invalid tree") from exc
            raise

        if len(response.nodes) != len(original_nodes):
            raise RuntimeError(
                f"Expected {len(original_nodes)} nodes from vectorizer, got {len(response.nodes)}"
            )

        vectorized_nodes: list[VectorizedSegmentNode] = []
        for orig_node, out_node in zip(original_nodes, response.nodes):
            content_chunks: list[tuple[str, np.ndarray]] = []
            for chunk in out_node.content:
                chunk_vec = _decode_vector(chunk.embedding)
                if chunk_vec is not None:
                    content_chunks.append((chunk.text, chunk_vec))

            vectorized_nodes.append(
                VectorizedSegmentNode(
                    name=orig_node.name,
                    level=orig_node.level,
                    embedding=_decode_vector(out_node.embedding),
                    content=content_chunks,
                    children=[],
                )
            )

        for idx, flat_in in enumerate(flat_inputs):
            start = flat_in.first_edge_idx
            end = start + flat_in.edge_count
            for edge_pos in range(start, end):
                child_idx = child_indices[edge_pos]
                vectorized_nodes[idx].children.append(vectorized_nodes[child_idx])

        return vectorized_nodes[0]
