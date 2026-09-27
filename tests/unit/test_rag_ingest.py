from pathlib import Path
from unittest.mock import Mock

from app.ai.rag.embeddings import EmbeddedChunk
from app.ai.rag.ingest import KnowledgeBaseIngestionService


def test_ingest_indexes_embedded_chunks(monkeypatch) -> None:
    directory = Path("knowledge_base")

    document = Mock()
    chunk = Mock()
    embedded_chunk = Mock(spec=EmbeddedChunk)

    monkeypatch.setattr(
        "app.ai.rag.ingest.load_documents",
        Mock(return_value=[document]),
    )

    monkeypatch.setattr(
        "app.ai.rag.ingest.chunk_documents",
        Mock(return_value=[chunk]),
    )

    embedding_service = Mock()
    embedding_service.embed_chunks.return_value = [embedded_chunk]

    vector_store = Mock()
    vector_store.upsert_chunks.return_value = 1

    service = KnowledgeBaseIngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = service.ingest(directory)

    assert result == 1

    embedding_service.embed_chunks.assert_called_once_with([chunk])
    vector_store.upsert_chunks.assert_called_once_with([embedded_chunk])