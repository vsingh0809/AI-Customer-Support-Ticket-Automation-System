from pathlib import Path

import pytest

from app.ai.rag.chunker import chunk_document
from app.ai.rag.embeddings import (
    EmbeddedChunk,
    EmbeddingError,
    EmbeddingService,
    FastEmbedProvider,
)
from app.ai.rag.loader import load_document

KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"


class FakeEmbeddingProvider:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)

        return [
            [float(index)] * self.dimension
            for index in range(len(texts))
        ]

    def embed_query(self,text:str)->list[float]:
        return[1.0]* self.dimension


class InvalidCountProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0]]


class InvalidDimensionsProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0], [1.0, 2.0, 3.0]]


def _get_chunks():
    document = load_document(KB_DIR / "refund_policy.md")

    return chunk_document(
        document,
        chunk_size=250,
        chunk_overlap=50,
    )


def test_embedding_service_preserves_chunk_order() -> None:
    chunks = _get_chunks()[:2]
    provider = FakeEmbeddingProvider()

    result = EmbeddingService(provider).embed_chunks(chunks)

    assert len(result) == 2
    assert [item.chunk.chunk_index for item in result] == [
        chunk.chunk_index for chunk in chunks
    ]


def test_embedding_service_returns_embedded_chunks() -> None:
    chunks = _get_chunks()[:2]
    provider = FakeEmbeddingProvider(dimension=4)

    result = EmbeddingService(provider).embed_chunks(chunks)

    assert all(isinstance(item, EmbeddedChunk) for item in result)
    assert all(len(item.embedding) == 4 for item in result)


def test_embedding_service_batches_requests() -> None:
    chunks = _get_chunks()[:3]
    provider = FakeEmbeddingProvider()

    result = EmbeddingService(
        provider,
        batch_size=2,
    ).embed_chunks(chunks)

    assert len(result) == 3
    assert len(provider.calls) == 2
    assert len(provider.calls[0]) == 2
    assert len(provider.calls[1]) == 1


def test_empty_chunks_returns_empty_result() -> None:
    provider = FakeEmbeddingProvider()

    result = EmbeddingService(provider).embed_chunks([])

    assert result == []
    assert provider.calls == []

def test_fastembed_provider_rejects_empty_text() -> None:
    provider = FastEmbedProvider(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=2,
    )

    with pytest.raises(EmbeddingError, match="empty text"):
        provider.embed_texts(["valid text", "   "])

def test_service_rejects_embedding_count_mismatch() -> None:
    chunks = _get_chunks()[:2]

    with pytest.raises(EmbeddingError, match="count mismatch"):
        EmbeddingService(InvalidCountProvider()).embed_chunks(chunks)


def test_service_rejects_inconsistent_dimensions() -> None:
    chunks = _get_chunks()[:2]

    with pytest.raises(EmbeddingError, match="inconsistent dimensions"):
        EmbeddingService(InvalidDimensionsProvider()).embed_chunks(chunks)


def test_fastembed_provider_initialization() -> None:
    provider = FastEmbedProvider(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=2,
    )

    assert provider is not None