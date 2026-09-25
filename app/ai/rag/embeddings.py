"""Embedding providers and embedding orchestration for the RAG pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from fastembed import TextEmbedding
from pydantic import BaseModel, ConfigDict, Field

from app.ai.rag.chunker import KnowledgeChunk


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class EmbeddingProvider(Protocol):
    """Provider contract used by the embedding service."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate one embedding vector for each input text."""


class EmbeddedChunk(BaseModel):
    """A knowledge chunk paired with its embedding vector."""

    model_config = ConfigDict(frozen=True)

    chunk: KnowledgeChunk
    embedding: tuple[float, ...] = Field(min_length=1)


class FastEmbedProvider:
    """Local embedding provider backed by FastEmbed."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
    ) -> None:
        if not model_name.strip():
            raise EmbeddingError("Embedding model name cannot be empty")

        if batch_size <= 0:
            raise EmbeddingError("batch_size must be greater than zero")

        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to initialize embedding model: {model_name}"
            ) from exc

        self._batch_size = batch_size

    def embed_query(self, text: str) -> list[float]:
        """Generate a retrieval query embedding."""
        normalized_texts = _validate_inputs([text])

        try:
         vectors = list(
            self._model.query_embed(normalized_texts[0])
        )
        except Exception as exc:
            raise EmbeddingError(
            "FastEmbed query embedding generation failed"
        ) from exc

        if len(vectors) != 1:
            raise EmbeddingError(
            f"Expected one query embedding, got {len(vectors)}"
        )

        embedding = vectors[0].astype(float).tolist()

        _validate_dimensions([embedding])

        return embedding
    
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:

        """Generate passage embeddings locally with FastEmbed."""
        normalized_texts = _validate_inputs(texts)

        try:
            vectors = list(
                self._model.passage_embed(normalized_texts)
            )
        except Exception as exc:
            raise EmbeddingError(
            "FastEmbed passage embedding generation failed"
            ) from exc

        embeddings = [
        vector.astype(float).tolist()
        for vector in vectors
         ]

        _validate_output_count(normalized_texts, embeddings)
        _validate_dimensions(embeddings)

        return embeddings



class EmbeddingService:
    """Convert knowledge chunks into embedded chunks."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be greater than zero")

        self._provider = provider
        self._batch_size = batch_size

    
    def embed_chunks(
        self,
        chunks: Sequence[KnowledgeChunk],
    ) -> list[EmbeddedChunk]:
        """Embed all chunks while preserving their original order."""
        if not chunks:
            return []

        embedded_chunks: list[EmbeddedChunk] = []

        for start in range(0, len(chunks), self._batch_size):
            batch = list(chunks[start : start + self._batch_size])
            texts = [chunk.content for chunk in batch]

            try:
                embeddings = self._provider.embed_texts(texts)
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(
                    "Unexpected embedding provider failure"
                ) from exc

            _validate_output_count(texts, embeddings)
            _validate_dimensions(embeddings)

            for chunk, embedding in zip(batch, embeddings, strict=True):
                embedded_chunks.append(
                    EmbeddedChunk(
                        chunk=chunk,
                        embedding=tuple(embedding),
                    )
                )

        return embedded_chunks

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a user search query."""
        if not query.strip():
            raise EmbeddingError("Query cannot be empty")

        try:
            embedding = self._provider.embed_query(query)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
            "Unexpected query embedding failure"
        ) from exc

        _validate_dimensions([embedding])

        return embedding

def _validate_inputs(texts: Sequence[str]) -> list[str]:
    """Validate and normalize embedding inputs."""
    if not texts:
        raise EmbeddingError("At least one text is required")

    normalized = [text.strip() for text in texts]

    if any(not text for text in normalized):
        raise EmbeddingError("Embedding input cannot contain empty text")

    return normalized


def _validate_output_count(
    inputs: Sequence[str],
    embeddings: Sequence[Sequence[float]],
) -> None:
    """Ensure one vector is returned for every input."""
    if len(inputs) != len(embeddings):
        raise EmbeddingError(
            f"Embedding count mismatch: expected {len(inputs)}, "
            f"got {len(embeddings)}"
        )


def _validate_dimensions(
    embeddings: Sequence[Sequence[float]],
) -> None:
    """Ensure all vectors have the same dimension."""
    if not embeddings:
        return

    dimensions = {len(embedding) for embedding in embeddings}

    if len(dimensions) != 1:
        raise EmbeddingError("Embedding vectors have inconsistent dimensions")

    if 0 in dimensions:
        raise EmbeddingError("Embedding vector cannot be empty")