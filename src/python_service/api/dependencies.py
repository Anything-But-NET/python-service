from fastapi import Request

from python_service.db.service import RagDbService
from python_service.services.parse_service import MarkdownParserService
from python_service.services.transform_service import MarkdownVectorizationService


def get_db_service(request: Request) -> RagDbService:
    return request.app.state.db_service


def get_vectorizer_service(request: Request) -> MarkdownVectorizationService:
    return request.app.state.vectorizer_service


def get_parser_service(request: Request) -> MarkdownParserService:
    return request.app.state.parser_service
