import pytest
import numpy as np
from unittest.mock import MagicMock

from python_service.services.query_filter_service import QueryFilterService, DEFAULT_SPAM_RESPONSE


@pytest.fixture
def mock_vectorizer():
    vectorizer = MagicMock()
    def fake_vectorize(text: str) -> np.ndarray:
        text_lower = text.lower()
        if any(w in text_lower for w in ["recipe", "sandwich", "weather", "food"]):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    vectorizer.vectorize_query.side_effect = fake_vectorize
    return vectorizer


def test_query_filter_blocks_out_of_domain_query(mock_vectorizer):
    """Out-of-domain spam query is identified as spam with default response."""
    service = QueryFilterService(vectorizer=mock_vectorizer)
    result = service.check_query("How to make a sandwich?")

    assert result.is_spam is True
    assert result.rejection_reason == "OUT_OF_DOMAIN"
    assert result.default_response == DEFAULT_SPAM_RESPONSE


def test_query_filter_allows_valid_query(mock_vectorizer):
    """Valid domain query passes filter."""
    service = QueryFilterService(vectorizer=mock_vectorizer)
    result = service.check_query("How to set up database connection?")

    assert result.is_spam is False
    assert result.default_response is None