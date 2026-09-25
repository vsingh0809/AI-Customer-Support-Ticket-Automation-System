"""Knowledge-base retrieval orchestration."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.ai.rag.embeddings import EmbeddingService
from app.ai.rag.vector_store import (
    QdrantVectorStore,
    VectorSearchResult,
)

DEFAULT_TOP_K: Final[int] = 5
DEFAULT_SCORE_THRESHOLD: Final[float] = 0.35


class RetrievalError(RuntimeError):
    """Raised when knowledge retrieval fails."""


class RetrievalConfig(BaseModel):
    """Configuration controlling retrieval behavior."""

    model_config = ConfigDict(frozen=True)

    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=20,
    )

    score_threshold: float = Field(
        default=DEFAULT_SCORE_THRESHOLD,
        ge=-1.0,
        le=1.0,
    )


class Retriever:
    """Retrieve relevant knowledge-base chunks for a query."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._config = config or RetrievalConfig()

    def retrieve(self, query: str) -> list[VectorSearchResult]:
        """Return relevant chunks for a user query."""
        if not query.strip():
            raise RetrievalError("Query cannot be empty")

        try:
            query_vector = self._embedding_service.embed_query(query)

            return self._vector_store.search(
                query_vector,
                limit=self._config.top_k,
                score_threshold=self._config.score_threshold,
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                "Knowledge-base retrieval failed"
            ) from exc