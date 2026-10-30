from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from python_service.models.domain import SegmentNode
from python_service.services.transform_service import MarkdownVectorizationService


def test_l2_normalize():
    zero_vec = np.array([0.0, 0.0, 0.0])
    norm_zero = MarkdownVectorizationService._l2_normalize(zero_vec)
    np.testing.assert_array_equal(zero_vec, norm_zero)

    vec = np.array([3.0, 4.0])
    norm_vec = MarkdownVectorizationService._l2_normalize(vec)
    np.testing.assert_almost_equal(np.linalg.norm(norm_vec), 1.0)
    np.testing.assert_almost_equal(norm_vec, [0.6, 0.8])


def test_merge_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    merged = MarkdownVectorizationService._merge_vectors([v1, v2], temperature=0.1)
    assert merged.shape == (3,)
    np.testing.assert_almost_equal(np.linalg.norm(merged), 1.0)


@pytest.fixture
def mock_vectorizer():
    with patch("python_service.services.transform_service.AutoTokenizer") as mock_tok_cls, \
         patch("python_service.services.transform_service.AutoModel") as mock_model_cls:

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Setup tokenizer behavior
        def fake_tokenizer(inputs, add_special_tokens=False, return_tensors=None, **kwargs):
            if isinstance(inputs, list):
                if return_tensors == "pt":
                    mock_res = MagicMock()
                    mock_res.to.return_value = {
                        "input_ids": torch.ones((len(inputs), 5), dtype=torch.long),
                        "attention_mask": torch.ones((len(inputs), 5), dtype=torch.long),
                    }
                    return mock_res
                return {"input_ids": [[1, 2, 3] for _ in inputs]}
            elif isinstance(inputs, str):
                if return_tensors == "pt":
                    return {
                        "input_ids": torch.ones((1, 8), dtype=torch.long),
                        "attention_mask": torch.ones((1, 8), dtype=torch.long),
                    }
                return {"input_ids": [1, 2, 3]}
            return {"input_ids": []}

        mock_tokenizer.side_effect = fake_tokenizer
        mock_tokenizer.decode.return_value = "decoded sentence chunk"

        # Setup model behavior
        # Return hidden state of shape (batch, seq_len, hidden_dim) where hidden_dim=4
        def fake_forward(**inputs):
            res = MagicMock()
            batch_size = inputs["input_ids"].shape[0] if "input_ids" in inputs else 1
            seq_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 8
            res.last_hidden_state = torch.ones((batch_size, seq_len, 4))
            return res

        mock_model.side_effect = fake_forward

        service = MarkdownVectorizationService(model_name="dummy/model")
        return service


def test_split_oversized_sentences(mock_vectorizer):
    sentences = ["short", "multiline\nsentence\nthat\nis\nlong", "long sentence without breaks"]

    # Test short sentences (<= max_chunk_size)
    mock_vectorizer.tokenizer.side_effect = lambda s, add_special_tokens=False: {
        "input_ids": [[1, 2] if s_item == "short" else [1] * 20 for s_item in s]
    }
    mock_vectorizer.tokenizer.decode.return_value = "sliced text"

    res = mock_vectorizer._split_oversized_sentences(
        sentences,
        max_chunk_size=5,
        chunk_overlap=1,
    )
    assert len(res) >= 3
    assert res[0] == ("short", 2)


def test_micro_split_chunks_subtopics_and_overflow(mock_vectorizer):
    # Empty text
    assert mock_vectorizer._micro_split_chunks("") == []
    assert mock_vectorizer._micro_split_chunks("   \n\t  ") == []

    mock_vectorizer.tokenizer.side_effect = lambda s, add_special_tokens=False: {
        "input_ids": [[1] * 10 for _ in s]
    }
    text = "First sentence. Second sentence. ** Subtopic header that is quite long enough."
    chunks = mock_vectorizer._micro_split_chunks(text, max_chunk_size=15, chunk_overlap=2)
    assert len(chunks) > 0


def test_get_embedding_empty_and_valid(mock_vectorizer):
    # Empty data
    assert mock_vectorizer._get_embedding("") == []
    assert mock_vectorizer._get_embedding("   ") == []

    # Valid data
    embeddings = mock_vectorizer._get_embedding("This is sample text for vectorization.")
    assert len(embeddings) > 0
    text, vec = embeddings[0]
    assert isinstance(text, str)
    assert isinstance(vec, np.ndarray)


def test_vectorize_query(mock_vectorizer):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        mock_vectorizer.vectorize_query("")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        mock_vectorizer.vectorize_query("   ")

    # Single or multi chunk query
    with patch.object(mock_vectorizer, "_get_embedding") as mock_get_emb:
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        mock_get_emb.return_value = [("query text", v1)]
        res = mock_vectorizer.vectorize_query("query text")
        np.testing.assert_array_equal(res, v1)

        v2 = np.array([0.0, 1.0, 0.0, 0.0])
        mock_get_emb.return_value = [("q1", v1), ("q2", v2)]
        res_multi = mock_vectorizer.vectorize_query("long query")
        assert res_multi.shape == (4,)


def test_vectorize_tree(mock_vectorizer):
    root = SegmentNode(name="Root", level=0)
    h1 = SegmentNode(name="Intro", level=1, content="Some introductory text.")
    h2 = SegmentNode(name="Empty Child", level=2, content="")
    h1.children.append(h2)
    root.children.append(h1)

    fake_vec = np.array([0.5, 0.5, 0.5, 0.5])
    with patch.object(mock_vectorizer, "_get_embedding", return_value=[("chunk", fake_vec)]):
        vec_root = mock_vectorizer.vectorize_tree(root)
        assert vec_root.name == "Root"
        assert vec_root.level == 0
        assert len(vec_root.children) == 1
        assert vec_root.children[0].name == "Intro"
        assert len(vec_root.children[0].children) == 1
