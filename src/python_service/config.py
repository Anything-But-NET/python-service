from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
    
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
    kafka_sync_status_topic: str = Field(
        default="rag.sync.status.v1",
        validation_alias=AliasChoices("KAFKA_SYNC_STATUS_TOPIC"),
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
    vectorizer_grpc_timeout: float = Field(
        default=30.0,
        validation_alias=AliasChoices("VECTORIZER_GRPC_TIMEOUT"),
    )
    vector_size: int = Field(
        default=1024,
        validation_alias=AliasChoices("VECTOR_SIZE"),
    )

    # OpenAI-compatible LLM settings
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("LLM_BASE_URL"),
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL"),
    )
    llm_timeout_seconds: float = Field(
        default=45.0,
        gt=0.0,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS"),
    )
    llm_max_tokens: int = Field(
        default=1200,
        ge=1,
        validation_alias=AliasChoices("LLM_MAX_TOKENS"),
    )

    # Grounded answer retrieval and evidence thresholds
    rag_retrieval_limit: int = Field(
        default=8,
        ge=1,
        le=100,
        validation_alias=AliasChoices("RAG_RETRIEVAL_LIMIT"),
    )
    rag_auto_merge_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("RAG_AUTO_MERGE_THRESHOLD"),
    )
    rag_min_relevance_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("RAG_MIN_RELEVANCE_SCORE"),
    )
    rag_min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("RAG_MIN_CONFIDENCE"),
    )
    rag_max_context_chars: int = Field(
        default=24000,
        ge=1000,
        validation_alias=AliasChoices("RAG_MAX_CONTEXT_CHARS"),
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


    @model_validator(mode="after")
    def validate_rag_thresholds(self) -> "Settings":
        if self.rag_min_relevance_score < self.rag_min_confidence:
            raise ValueError(
                "RAG_MIN_RELEVANCE_SCORE must be greater than or equal to "
                "RAG_MIN_CONFIDENCE to avoid sending evidence that cannot pass "
                "the final confidence gate."
            )
        return self


@lru_cache
def get_settings() -> Settings:

    return Settings()
