from unittest.mock import Mock
from uuid import UUID

import pytest

from app.ai.rag.chunker import chunk_document
from app.ai.rag.embeddings import EmbeddedChunk
from app.ai.rag.loader import load_document
from app.ai.rag.vector_store import (
    QdrantVectorStore,
    VectorSearchResult,
    VectorStoreError,
)


def _build_embedded_chunk(
    index: int = 0,
    dimension: int = 3,
) -> EmbeddedChunk:
    document = load_document("knowledge_base/faq.md")

    chunks = chunk_document(
        document,
        chunk_size=500,
        chunk_overlap=50,
    )

    chunk = chunks[index]

    return EmbeddedChunk(
        chunk=chunk,
        embedding=tuple([0.1] * dimension),
    )


def test_ensure_collection_creates_missing_collection() -> None:
    client = Mock()
    client.collection_exists.return_value = False

    store = QdrantVectorStore(
        client,
        collection_name="support_kb",
        vector_size=3,
    )

    store.ensure_collection()

    client.create_collection.assert_called_once()

    args = client.create_collection.call_args.kwargs

    assert args["collection_name"] == "support_kb"
    assert args["vectors_config"].size == 3


def test_ensure_collection_does_not_recreate_existing_collection() -> None:
    client = Mock()
    client.collection_exists.return_value = True

    store = QdrantVectorStore(
        client,
        collection_name="support_kb",
        vector_size=3,
    )

    store.ensure_collection()

    client.create_collection.assert_not_called()


def test_upsert_empty_chunks_returns_zero() -> None:
    client = Mock()

    store = QdrantVectorStore(
        client,
        collection_name="support_kb",
        vector_size=3,
    )

    result = store.upsert_chunks([])

    assert result == 0
    client.upsert.assert_not_called()


def test_upsert_chunks_writes_points() -> None:
    client = Mock()
    client.collection_exists.return_value = True

    store = QdrantVectorStore(
        client,
        collection_name="support_kb",
        vector_size=3,
    )

    embedded_chunk = _build_embedded_chunk()

    result = store.upsert_chunks([embedded_chunk])

    assert result == 1
    client.upsert.assert_called_once()

    kwargs = client.upsert.call_args.kwargs
    points = kwargs["points"]

    assert len(points) == 1
    assert isinstance(points[0].id, UUID)
    assert points[0].payload["chunk_id"] == embedded_chunk.chunk.chunk_id
    assert points[0].payload["document_id"] == embedded_chunk.chunk.document_id


def test_upsert_rejects_wrong_vector_dimension() -> None:
    client = Mock()

    store = QdrantVectorStore(
        client,
        collection_name="support_kb",
        vector_size=3,
    )

    embedded_chunk = _build_embedded_chunk(dimension=2)

    with pytest.raises(
        VectorStoreError,
        match="Invalid vector dimension",
    ):
        store.upsert_chunks([embedded_chunk])


def test_search_rejects_wrong_query_dimension() -> None:
    client = Mock()

    store = QdrantVectorStore(
        client,
        collection_name="support_kb",
        vector_size=3,
    )

    with pytest.raises(
        VectorStoreError,
        match="Invalid query vector dimension",
    ):
        store.search([0.1, 0.2])


def test_search_maps_qdrant_results() -> None:
    client = Mock()

    point = Mock()
    point.score = 0.92
    point.payload = {
        "chunk_id": "refund-policy-chunk-000",
        "document_id": "refund-policy",
        "title": "Refund Policy",
        "category": "refund",
        "version": "1.0",
        "source": "refund_policy.md",
        "chunk_index": 0,
        "total_chunks": 2,
        "content": "Customers may request refunds within 7 days.",
    }

    response = Mock()
    response.points = [point]

    client.query_points.return_value = response

    store = QdrantVectorStore(
        client,
        collection_name="support_kb",
        vector_size=3,
    )

    results = store.search(
        [0.1, 0.2, 0.3],
        limit=5,
    )

    assert len(results) == 1
    assert isinstance(results[0], VectorSearchResult)
    assert results[0].chunk_id == "refund-policy-chunk-000"
    assert results[0].score == 0.92
    assert results[0].metadata["category"] == "refund"