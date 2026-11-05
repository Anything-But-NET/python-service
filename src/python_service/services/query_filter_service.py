import numpy as np
from dataclasses import dataclass
from typing import List
from enum import Enum

from python_service.services.transform_service import VectorizationService

DEFAULT_SPAM_RESPONSE = "I can only answer work-related questions based on company documentation."

SPAM_REFERENCE_PHRASES = [
    "food recipe cooking sandwich weather joke poem election politics"
]

class RejectionReason(str, Enum):
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two 1D vectors."""
    norm_1 = np.linalg.norm(v1)
    norm_2 = np.linalg.norm(v2)
    if norm_1 == 0 or norm_2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm_1 * norm_2))


@dataclass
class FilterResult:
    is_spam: bool
    rejection_reason: RejectionReason | None = None
    default_response: str | None = None


class QueryFilterService:
    """Service responsible for filtering out-of-domain and inappropriate queries using semantic similarity."""

    def __init__(self, vectorizer: VectorizationService, threshold: float = 0.6):
        self.vectorizer = vectorizer
        self.threshold = threshold
        self._spam_vectors: List[np.ndarray] = [
            self.vectorizer.vectorize_query(phrase)
            for phrase in SPAM_REFERENCE_PHRASES
        ]

    def check_query(self, query: str) -> FilterResult:
        """Check if query is out-of-domain or spam."""
        query_vec = self.vectorizer.vectorize_query(query)

        for spam_vec in self._spam_vectors:
            similarity = cosine_similarity(query_vec, spam_vec)
            if similarity >= self.threshold:
                return FilterResult(
                    is_spam=True,
                    rejection_reason=RejectionReason.OUT_OF_DOMAIN,
                    default_response=DEFAULT_SPAM_RESPONSE,
                )

        return FilterResult(is_spam=False)