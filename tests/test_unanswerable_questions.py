import pytest
from python_service.retrieval_service import UserContext
from python_service.services.qa_service import QAService


@pytest.fixture
def qa_service():
    return QAService(vectorizer=None)  # Мок без gRPC для pytest


def test_answerable_question_returns_factual_answer_and_citations(qa_service):
    """AC 1: Answerable questions receive factual answers with citations."""
    user = UserContext(user_id="usr_dev_1", groups=["developer"])
    response = qa_service.answer_question("Backend Python", user)

    assert response.status == "ANSWERED"
    assert "FastAPI" in response.answer
    assert len(response.citations) > 0
    assert response.create_ticket is False


def test_out_of_domain_spam_question_rejected_without_ticket(qa_service):
    """AC 1 & AC 2: Out-of-domain questions are rejected without creating tickets."""
    user = UserContext(user_id="usr_dev_1", groups=["developer"])
    response = qa_service.answer_question("How can I prepare a sandwich?", user)

    assert response.status == "SPAM_REJECTED"
    assert response.create_ticket is False


def test_missing_knowledge_question_escalates_and_creates_ticket(qa_service):
    """AC 2: Unanswerable work questions trigger escalation and ticket creation."""
    user = UserContext(user_id="usr_dev_1", groups=["developer"])
    response = qa_service.answer_question("VPN configuration guidelines", user)

    assert response.status == "MISSING_KNOWLEDGE"
    assert response.create_ticket is True