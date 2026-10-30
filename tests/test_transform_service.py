from concurrent import futures

import grpc
import numpy as np
import pytest

from python_service.grpc_gen import vectorizer_pb2, vectorizer_pb2_grpc
from python_service.models.domain import SegmentNode
from python_service.services.transform_service import (
    VectorizationService,
    _decode_vector,
)


class FakeVectorizerServicer(vectorizer_pb2_grpc.MarkdownVectorizerServicer):
    def VectorizeQuery(self, request, context):
        if not request.query.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Query cannot be empty")
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return vectorizer_pb2.QueryResponse(embedding=vec.tobytes())

    def VectorizeTree(self, request, context):
        out_nodes = []
        for node in request.nodes:
            vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            chunks = []
            if node.content.strip():
                chunks.append(
                    vectorizer_pb2.VectorizedChunk(
                        text=node.content.strip(),
                        embedding=vec.tobytes(),
                    )
                )
            out_nodes.append(
                vectorizer_pb2.FlatNodeOutput(
                    embedding=vec.tobytes() if (node.content.strip() or node.edge_count > 0 or node.level > 0) else b"",
                    content=chunks,
                )
            )
        return vectorizer_pb2.TreeResponse(nodes=out_nodes)


@pytest.fixture
def grpc_vectorizer_service():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    vectorizer_pb2_grpc.add_MarkdownVectorizerServicer_to_server(
        FakeVectorizerServicer(), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    service = VectorizationService(grpc_target=f"127.0.0.1:{port}")
    try:
        yield service
    finally:
        service.close()
        server.stop(grace=None)


def test_decode_vector():
    assert _decode_vector(b"") is None
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    decoded = _decode_vector(vec.tobytes())
    assert decoded is not None
    np.testing.assert_array_equal(decoded, vec)


def test_vectorize_query(grpc_vectorizer_service):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        grpc_vectorizer_service.vectorize_query("")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        grpc_vectorizer_service.vectorize_query("   ")

    res = grpc_vectorizer_service.vectorize_query("query text")
    np.testing.assert_array_equal(res, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_vectorize_tree(grpc_vectorizer_service):
    root = SegmentNode(name="Root", level=0)
    h1 = SegmentNode(name="Intro", level=1, content="Some introductory text.")
    h2 = SegmentNode(name="Empty Child", level=2, content="")
    h1.children.append(h2)
    root.children.append(h1)

    vec_root = grpc_vectorizer_service.vectorize_tree(root)
    assert vec_root.name == "Root"
    assert vec_root.level == 0
    assert vec_root.embedding is not None
    assert len(vec_root.children) == 1

    vec_h1 = vec_root.children[0]
    assert vec_h1.name == "Intro"
    assert vec_h1.level == 1
    assert vec_h1.embedding is not None
    assert len(vec_h1.content) == 1
    assert vec_h1.content[0][0] == "Some introductory text."

    assert len(vec_h1.children) == 1
    vec_h2 = vec_h1.children[0]
    assert vec_h2.name == "Empty Child"
    assert vec_h2.level == 2
    assert vec_h2.embedding is not None
    assert vec_h2.content == []
