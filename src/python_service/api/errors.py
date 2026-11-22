from python_service.api.schemas import DashboardErrorCode, GenerationErrorStatus


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


class DashboardApiError(RuntimeError):
    """Structured error for Web Dashboard endpoints (/api/v1/dash/*)."""

    def __init__(
        self,
        http_status: int,
        error_code: DashboardErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.error_code = error_code
        self.message = message
