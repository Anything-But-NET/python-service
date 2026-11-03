from dataclasses import dataclass
from typing import List, Dict, Any
from python_service.retrieval_service import RetrievalResponse, retrieve_mock_results, UserContext

SPAM_KEYWORDS = [
    "sandwich", "recipe", "weather", "poem", "election",
    "presidential", "joke", "song", "translate"
]

DEFAULT_SPAM_RESPONSE = "I can only answer work-related questions based on company documentation."
DEFAULT_MISSING_KNOWLEDGE_RESPONSE = "I couldn't find information about this in the knowledge base. I have created a support ticket for you."


@dataclass
class QAResponse:
    answer: str
    status: str  # "ANSWERED", "SPAM_REJECTED", "MISSING_KNOWLEDGE"
    citations: List[Dict[str, Any]]
    create_ticket: bool


def is_spam_or_out_of_domain(query: str) -> bool:
    """Check if query is out-of-domain spam based on keywords."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in SPAM_KEYWORDS)


def answer_question(query: str, user: UserContext) -> QAResponse:
    """
    Safely process question and return appropriate response based on retrieval and domain checks.
    """
    if is_spam_or_out_of_domain(query):
        return QAResponse(
            answer=DEFAULT_SPAM_RESPONSE,
            status="SPAM_REJECTED",
            citations=[],
            create_ticket=False,
        )

    retrieval_res: RetrievalResponse = retrieve_mock_results(query, user)

    if not retrieval_res.chunks:
        return QAResponse(
            answer=DEFAULT_MISSING_KNOWLEDGE_RESPONSE,
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