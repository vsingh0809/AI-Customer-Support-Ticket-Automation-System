"""Qdrant vector-store integration for knowledge-base embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models

from app.ai.rag.embeddings import EmbeddedChunk


class VectorStoreError(RuntimeError):
    """Raised when vector-store operations fail."""


class VectorSearchResult(BaseModel):
    """A vector-search result returned from Qdrant."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float
    metadata: dict[str, str | int]


class QdrantVectorStore:
    """Store and search knowledge-base embeddings in Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = "support_kb",
        vector_size: int = 384,
    ) -> None:
        if not collection_name.strip():
            raise VectorStoreError("Collection name cannot be empty")

        if vector_size <= 0:
            raise VectorStoreError(
                "Vector size must be greater than zero"
            )

        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size

    def ensure_collection(self) -> None:
        """Create the collection when it does not already exist."""
        try:
            exists = self._client.collection_exists(
                collection_name=self._collection_name
            )

            if exists:
                return

            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize collection: {self._collection_name}"
            ) from exc

    def upsert_chunks(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> int:
        """Insert or update embedded chunks in Qdrant."""
        if not embedded_chunks:
            return 0

        points: list[models.PointStruct] = []

        for item in embedded_chunks:
            embedding = list(item.embedding)

            if len(embedding) != self._vector_size:
                raise VectorStoreError(
                    f"Invalid vector dimension for chunk "
                    f"{item.chunk.chunk_id}: "
                    f"expected {self._vector_size}, got {len(embedding)}"
                )

            chunk = item.chunk

            point_id = _point_id_from_chunk_id(chunk.chunk_id)

            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "category": chunk.category,
                "version": chunk.version,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "content": chunk.content,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        try:
            self.ensure_collection()

            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                "Failed to upsert knowledge-base chunks"
            ) from exc

        return len(points)

    def search(
    self,
    query_vector: Sequence[float],
    *,
    limit: int = 5,
    score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search Qdrant for the most similar knowledge chunks."""
        if not query_vector:
            raise VectorStoreError("Query vector cannot be empty")

        if limit <= 0:
            raise VectorStoreError(
            "Search limit must be greater than zero"
        )

        if len(query_vector) != self._vector_size:
            raise VectorStoreError(
            f"Invalid query vector dimension: "
            f"expected {self._vector_size}, "
            f"got {len(query_vector)}"
        )

        if score_threshold is not None and not -1.0 <= score_threshold <= 1.0:
            raise VectorStoreError(
            "Cosine score threshold must be between -1 and 1"
        )

        try:
            response = self._client.query_points(
            collection_name=self._collection_name,
            query=list(query_vector),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        except Exception as exc:
            raise VectorStoreError(
            "Failed to search Qdrant"
        ) from exc

        results: list[VectorSearchResult] = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                VectorSearchResult(
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                content=str(payload["content"]),
                score=float(point.score),
                metadata={
                    "title": str(payload["title"]),
                    "category": str(payload["category"]),
                    "version": str(payload["version"]),
                    "source": str(payload["source"]),
                    "chunk_index": int(payload["chunk_index"]),
                    "total_chunks": int(payload["total_chunks"]),
                },
            )
        )

        return results


def _point_id_from_chunk_id(chunk_id: str) -> UUID:
    """Create a deterministic UUID accepted by Qdrant."""
    return uuid5(NAMESPACE_URL, chunk_id)