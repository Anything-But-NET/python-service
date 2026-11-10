from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python_service.services.parse_service import MarkdownParserService
    from python_service.services.transform_service import VectorizationService

__all__ = ["MarkdownParserService", "VectorizationService"]


def __getattr__(name: str):
    if name == "MarkdownParserService":
        from python_service.services.parse_service import MarkdownParserService

        return MarkdownParserService
    if name == "VectorizationService":
        from python_service.services.transform_service import VectorizationService

        return VectorizationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
