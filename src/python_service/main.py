import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
import logging

from fastapi import FastAPI
import uvicorn

from python_service.api.routes import router
from python_service.config import get_settings
from python_service.db.service import RagDbService
from python_service.kafka.consumer import IngestionKafkaWorker
from python_service.services.parse_service import MarkdownParserService
from python_service.services.transform_service import MarkdownVectorizationService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Initializing services...")

    parser_service = MarkdownParserService()
    vectorizer_service = await asyncio.to_thread(
        MarkdownVectorizationService,
        settings.embedding_model_name,
    )
    db_service = RagDbService(db_url=settings.db_url)
    await db_service.init_schema()

    app.state.parser_service = parser_service
    app.state.vectorizer_service = vectorizer_service
    app.state.db_service = db_service

    kafka_task: asyncio.Task[None] | None = None
    kafka_worker: IngestionKafkaWorker | None = None

    if settings.kafka_enabled:
        kafka_worker = IngestionKafkaWorker(
            parser=parser_service,
            vectorizer=vectorizer_service,
            db=db_service,
            kafka_brokers=settings.kafka_brokers,
            group_id=settings.kafka_group_id,
            topic=settings.kafka_topic,
        )
        kafka_task = asyncio.create_task(
            kafka_worker.start_consuming(),
            name="kafka-ingestion-worker",
        )

    try:
        yield
    finally:
        logger.info("Shutting down services...")
        if kafka_worker is not None:
            kafka_worker.stop()
        if kafka_task is not None:
            kafka_task.cancel()
            with suppress(asyncio.CancelledError):
                await kafka_task
        await db_service.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Python RAG Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "python_service.main:app",
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
