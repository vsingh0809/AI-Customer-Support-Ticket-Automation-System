"""Configuration for the Streamlit customer-support frontend."""

from functools import lru_cache
from uuid import UUID

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    """Runtime configuration required by the Streamlit frontend."""

    backend_api_url: str = Field(
        default="http://localhost:8000",
        min_length=1,
    )

    demo_customer_id: UUID

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> FrontendSettings:
    """Return cached frontend settings."""
    return FrontendSettings()