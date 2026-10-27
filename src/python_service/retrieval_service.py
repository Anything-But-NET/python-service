from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- 1. Моделі даних  ---

@dataclass
class UserContext:
    user_id: str
    groups: List[str] = field(default_factory=list)  # Передається як user_permissions.groups

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    page_number: int

@dataclass
class Citation:
    title: str  
    url: str   
    document_id: Optional[str] = None

@dataclass
class RetrievalResponse:
    chunks: List[Chunk]
    citations: List[Citation]

@dataclass
class MockDocument:
    document_id: str
    title: str
    url: str
    allowed_groups: List[str] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)


# --- 2. Мокові дані документів ---

MOCK_DOCUMENTS_DB: List[MockDocument] = [
    MockDocument(
        document_id="doc_backend_001",
        title="Backend Architecture Guidelines",
        url="https://mock-sharepoint/docs/backend_architecture.pdf",
        allowed_groups=["developer", "tech_lead", "usr_dev_1"],
        chunks=[
            {"chunk_id": "chunk_001", "text": "Backend architecture uses Python and FastAPI...", "page_number": 1},
            {"chunk_id": "chunk_002", "text": "PostgreSQL is used for storing metadata...", "page_number": 2}
        ]
    ),
    MockDocument(
        document_id="doc_finance_999",
        title="Financial Report 2026",
        url="https://mock-sharepoint/docs/financial_report_2026.pdf",
        allowed_groups=["finance", "c_level", "usr_cfo_1"],
        chunks=[
            {"chunk_id": "chunk_003", "text": "Confidential budget details for 2026...", "page_number": 1}
        ]
    )
]


# --- 3. Логіка перевірки доступу та фільтрації ---

def has_access(user: UserContext, doc: MockDocument) -> bool:
    """Перевірка чи має користувач хоча б одну спільну групу з allowed_groups документа."""
    return any(group in doc.allowed_groups for group in user.groups)


def retrieve_mock_results(query: str, user: UserContext) -> RetrievalResponse:
    """Основна функція retrieval з фільтрацією за контрактом groups."""
    result_chunks: List[Chunk] = []
    result_citations: List[Citation] = []

    for doc in MOCK_DOCUMENTS_DB:
        # Фільтрація документів за allowed_groups
        if not has_access(user, doc):
            continue

        matching_chunks = [
            c for c in doc.chunks if any(word in c["text"].lower() for word in query.lower().split())
        ]

        if matching_chunks:
            # Цитата створюється за контрактом з title та url
            result_citations.append(
                Citation(
                    title=doc.title,
                    url=doc.url,
                    document_id=doc.document_id
                )
            )

            for c in matching_chunks:
                result_chunks.append(
                    Chunk(
                        chunk_id=c["chunk_id"],
                        document_id=doc.document_id,
                        text=c["text"],
                        page_number=c["page_number"]
                    )
                )

    return RetrievalResponse(chunks=result_chunks, citations=result_citations)