import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from python_service.retrieval_service import RetrievalResponse, retrieve_mock_results, UserContext
from python_service.services.transform_service import VectorizationService

# 1. Шаблони відповідей винесені з коду (Інтернаціоналізація / конфіг)
RESPONSE_TEMPLATES = {
    "SPAM_REJECTED": "I can only answer work-related questions based on company documentation.",
    "MISSING_KNOWLEDGE": "I couldn't find information about this in the knowledge base. I have created a support ticket for you.",
}

# 2. Фрази для семантичної фільтрації (Semantic Routing)
DOMAIN_REFERENCE_PHRASE = "software engineering backend architecture database corporate guidelines"
SPAM_REFERENCE_PHRASES = [
    "food recipe cooking sandwich weather joke poem election politics"
]


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two 1D vectors."""
    norm_1 = np.linalg.norm(v1)
    norm_2 = np.linalg.norm(v2)
    if norm_1 == 0 or norm_2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm_1 * norm_2))


@dataclass
class QAResponse:
    answer: str
    status: str  # "ANSWERED", "SPAM_REJECTED", "MISSING_KNOWLEDGE"
    citations: List[Dict[str, Any]]
    create_ticket: bool


class QAService:
    """OOP Service for QA handling and semantic query routing."""

    def __init__(self, vectorizer: Optional[VectorizationService] = None):
        self.vectorizer = vectorizer
        self._spam_vectors: List[np.ndarray] = []
        
        # Передустановлення векторів для офф-домен фільтрації (якщо векторйзер доступний)
        if self.vectorizer:
            for phrase in SPAM_REFERENCE_PHRASES:
                try:
                    self._spam_vectors.append(self.vectorizer.vectorize_query(phrase))
                except Exception:
                    pass

    def is_spam_semantic(self, query: str, threshold: float = 0.6) -> bool:
        """Check if query is spam using cosine similarity over embeddings."""
        if not self.vectorizer or not self._spam_vectors:
            # Fallback для тестового середовища без gRPC векторйзера
            spam_words = ["sandwich", "recipe", "weather", "poem", "election", "joke"]
            return any(w in query.lower() for w in spam_words)

        query_vec = self.vectorizer.vectorize_query(query)
        for spam_vec in self._spam_vectors:
            similarity = cosine_similarity(query_vec, spam_vec)
            if similarity >= threshold:
                return True
        return False

    def answer_question(self, query: str, user: UserContext) -> QAResponse:
        """Process user question and route to appropriate flow."""
        if self.is_spam_semantic(query):
            return QAResponse(
                answer=RESPONSE_TEMPLATES["SPAM_REJECTED"],
                status="SPAM_REJECTED",
                citations=[],
                create_ticket=False,
            )

        retrieval_res: RetrievalResponse = retrieve_mock_results(query, user)

        if not retrieval_res.chunks:
            return QAResponse(
                answer=RESPONSE_TEMPLATES["MISSING_KNOWLEDGE"],
                status="MISSING_KNOWLEDGE",
                citations=[],
                create_ticket=True,
            )

        context_text = "\n".join([c.text for c in retrieval_res.chunks])
        citations_data = [
            {"title": c.title, "url": c.url} for c in retrieval_res.citations
        ]

        return QAResponse(
            answer=f"Based on documentation: {context_text}",
            status="ANSWERED",
            citations=citations_data,
            create_ticket=False,
        )