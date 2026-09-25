from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "customer-support-ai"
    database_url: str = "postgresql+psycopg://support_user:support_password@localhost:5432/support_db"

    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = None
    embedding_batch_size: int = 64

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "support_kb"
    qdrant_vector_size: int = 384
    qdrant_timeout: float = 10.0

    deepseek_api_key: str | None = None
    llm_model: str = "deepseek-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
