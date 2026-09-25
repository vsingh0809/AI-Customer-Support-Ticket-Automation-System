from unittest.mock import Mock

import pytest

from app.ai.rag.embeddings import EmbeddingService
from app.ai.rag.retriever import (
    RetrievalConfig,
    RetrievalError,
    Retriever,
)
from app.ai.rag.vector_store import VectorSearchResult


class FakeEmbeddingProvider:
    def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def _result() -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id="refund-policy-chunk-000",
        document_id="refund-policy",
        content="Refunds are available within 7 days.",
        score=0.91,
        metadata={
            "title": "Refund Policy",
            "category": "refund",
            "version": "1.0",
            "source": "refund_policy.md",
            "chunk_index": 0,
            "total_chunks": 2,
        },
    )


def test_retriever_embeds_query_and_searches_vector_store() -> None:
    embedding_service = EmbeddingService(
        FakeEmbeddingProvider()
    )

    vector_store = Mock()
    vector_store.search.return_value = [_result()]

    retriever = Retriever(
        embedding_service,
        vector_store,
    )

    results = retriever.retrieve(
        "What is the refund policy?"
    )

    assert len(results) == 1
    assert results[0].score == 0.91

    vector_store.search.assert_called_once_with(
        [1.0, 0.0, 0.0],
        limit=5,
        score_threshold=0.35,
    )


def test_retriever_uses_custom_configuration() -> None:
    embedding_service = EmbeddingService(
        FakeEmbeddingProvider()
    )

    vector_store = Mock()
    vector_store.search.return_value = []

    config = RetrievalConfig(
        top_k=10,
        score_threshold=0.7,
    )

    retriever = Retriever(
        embedding_service,
        vector_store,
        config,
    )

    retriever.retrieve("Refund eligibility")

    vector_store.search.assert_called_once_with(
        [1.0, 0.0, 0.0],
        limit=10,
        score_threshold=0.7,
    )


def test_retriever_returns_empty_list_when_nothing_is_relevant() -> None:
    embedding_service = EmbeddingService(
        FakeEmbeddingProvider()
    )

    vector_store = Mock()
    vector_store.search.return_value = []

    retriever = Retriever(
        embedding_service,
        vector_store,
    )

    results = retriever.retrieve(
        "Completely unrelated question"
    )

    assert results == []


def test_retriever_rejects_empty_query() -> None:
    embedding_service = EmbeddingService(
        FakeEmbeddingProvider()
    )

    vector_store = Mock()

    retriever = Retriever(
        embedding_service,
        vector_store,
    )

    with pytest.raises(
        RetrievalError,
        match="Query cannot be empty",
    ):
        retriever.retrieve("   ")

    vector_store.search.assert_not_called()


def test_retrieval_configuration_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        RetrievalConfig(top_k=0)

    with pytest.raises(ValueError):
        RetrievalConfig(top_k=21)


def test_retrieval_configuration_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        RetrievalConfig(score_threshold=1.1)

    with pytest.raises(ValueError):
        RetrievalConfig(score_threshold=-1.1)