from python_service.api.schemas import GenerationErrorStatus


class GenerationServiceError(RuntimeError):
    """Expected infrastructure failure while generating a grounded answer."""

    def __init__(
        self,
        status: GenerationErrorStatus,
        message: str,
        *,
        query_id: str | None = None,
        http_status: int = 503,
        description: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.query_id = query_id
        self.http_status = http_status
        self.description = description
