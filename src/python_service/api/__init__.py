__all__ = ["router"]


def __getattr__(name: str):
    if name == "router":
        from python_service.api.routes import router

        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
