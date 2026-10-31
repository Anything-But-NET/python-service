from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database settings
    db_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5433/postgres",
        validation_alias=AliasChoices("DB_URL", "VECTOR_DB_URL", "DATABASE_URL"),
    )

    # Kafka settings
    kafka_brokers: str = Field(
        default="localhost:9092",
        validation_alias=AliasChoices("KAFKA_BROKERS", "KAFKA_BROKER"),
    )
    kafka_group_id: str = Field(
        default="rag-ingestion-group",
        validation_alias=AliasChoices("KAFKA_GROUP_ID"),
    )
    kafka_topic: str = Field(
        default="rag.documents.ingestion.v1",
        validation_alias=AliasChoices("KAFKA_TOPIC"),
    )
    kafka_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("KAFKA_ENABLED"),
    )

    # Vectorizer gRPC settings
    vectorizer_grpc_target: str = Field(
        default="localhost:50051",
        validation_alias=AliasChoices(
            "VECTORIZER_GRPC_TARGET",
            "VECTORIZER_GRPC_URL",
            "VECTORIZER_URL",
        ),
    )
    vector_size: int = Field(
        default=1024,
        validation_alias=AliasChoices("VECTOR_SIZE"),
    )

    # FastAPI / Uvicorn settings
    api_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("API_HOST"),
    )
    api_port: int = Field(
        default=8000,
        validation_alias=AliasChoices("API_PORT"),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
