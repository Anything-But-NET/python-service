import sys
import os
sys.path.append(os.path.dirname(__file__))

from retrieval_service import (
    retrieve_mock_results, 
    UserContext
)

def test_allowed_user_receives_chunks_and_citations():
    """Перевірка сценарію: Allowed User з потрібною групою отримує результати."""
    allowed_user = UserContext(
        user_id="usr_dev_1",
        groups=["developer"]
    )

    response = retrieve_mock_results(query="Backend Python", user=allowed_user)

    assert len(response.chunks) > 0
    assert len(response.citations) > 0
    assert response.citations[0].title == "Backend Architecture Guidelines"
    assert response.citations[0].url == "https://mock-sharepoint/docs/backend_architecture.pdf"
    assert response.chunks[0].document_id == "doc_backend_001"


def test_denied_user_receives_no_restricted_content():
    """Перевірка сценарію: Denied User без відповідної групи НЕ отримує секретні дані."""
    unauthorized_user = UserContext(
        user_id="usr_dev_1",
        groups=["developer"] 
    )

    response = retrieve_mock_results(query="budget financial", user=unauthorized_user)

    assert len(response.chunks) == 0
    assert len(response.citations) == 0


def test_partially_allowed_user_gets_only_authorized_docs():
    """Перевірка сценарію: повертаються лише документи дозволених груп."""
    dev_user = UserContext(
        user_id="usr_dev_1",
        groups=["developer"]
    )

    response = retrieve_mock_results(query="for", user=dev_user)

    titles = [citation.title for citation in response.citations]
    assert "Backend Architecture Guidelines" in titles
    assert "Financial Report 2026" not in titles