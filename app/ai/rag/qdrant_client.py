"""Qdrant client construction."""

from qdrant_client import QdrantClient

from app.core.config import get_settings


def get_qdrant_client() -> QdrantClient:
    """Create an authenticated Qdrant Cloud client."""
    settings = get_settings()

    if not settings.qdrant_url:
        raise ValueError("QDRANT_URL is not configured")

    if not settings.qdrant_api_key:
        raise ValueError("QDRANT_API_KEY is not configured")

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )