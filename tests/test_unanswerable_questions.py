import pytest
from python_service.retrieval_service import UserContext
from python_service.services.qa_service import answer_question


def test_answerable_question_returns_factual_answer_and_citations():
    """AC 1: Answerable questions receive factual answers with citations."""
    user = UserContext(user_id="usr_dev_1", groups=["developer"])
    response = answer_question("Backend Python", user)

    assert response.status == "ANSWERED"
    assert "FastAPI" in response.answer
    assert len(response.citations) > 0
    assert response.create_ticket is False


def test_out_of_domain_spam_question_rejected_without_ticket():
    """AC 1 & AC 2: Out-of-domain questions are rejected without creating tickets."""
    user = UserContext(user_id="usr_dev_1", groups=["developer"])
    response = answer_question("How can I prepare a sandwich?", user)

    assert response.status == "SPAM_REJECTED"
    assert response.answer == "I can only answer work-related questions based on company documentation."
    assert len(response.citations) == 0
    assert response.create_ticket is False


def test_missing_knowledge_question_escalates_and_creates_ticket():
    """AC 2: Unanswerable work questions trigger escalation and ticket creation."""
    user = UserContext(user_id="usr_dev_1", groups=["developer"])
    
    response = answer_question("VPN configuration guidelines", user)

    assert response.status == "MISSING_KNOWLEDGE"
    assert response.answer == "I couldn't find information about this in the knowledge base. I have created a support ticket for you."
    assert len(response.citations) == 0
    assert response.create_ticket is True


def test_unauthorized_access_results_in_missing_knowledge_escalation():
    """AC 2: Restricted content requests do not leak info and trigger escalation."""
    user = UserContext(user_id="usr_dev_1", groups=["developer"])
    response = answer_question("Financial Report 2026 budget", user)

    assert response.status == "MISSING_KNOWLEDGE"
    assert response.create_ticket is True