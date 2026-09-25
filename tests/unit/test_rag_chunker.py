from pathlib import Path

import pytest

from app.ai.rag.chunker import ChunkingError, chunk_document, chunk_documents
from app.ai.rag.loader import load_document

KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"


def test_small_document_creates_single_chunk() -> None:
    document = load_document(KB_DIR / "refund_policy.md")

    chunks = chunk_document(document, chunk_size=3200, chunk_overlap=400)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1
    assert chunks[0].content
    assert chunks[0].document_id == document.metadata.document_id


def test_large_document_creates_multiple_chunks() -> None:
    document = load_document(KB_DIR / "refund_policy.md")

    chunks = chunk_document(document, chunk_size=250, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(chunk.content for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunks_preserve_metadata() -> None:
    document = load_document(KB_DIR / "shipping_policy.md")

    chunks = chunk_document(document, chunk_size=250, chunk_overlap=50)

    for chunk in chunks:
        assert chunk.document_id == document.metadata.document_id
        assert chunk.title == document.metadata.title
        assert chunk.category == document.metadata.category
        assert chunk.version == document.metadata.version
        assert chunk.source == document.metadata.source
        assert chunk.total_chunks == len(chunks)


def test_chunks_have_deterministic_ids() -> None:
    document = load_document(KB_DIR / "payment_policy.md")

    first = chunk_document(document, chunk_size=250, chunk_overlap=50)
    second = chunk_document(document, chunk_size=250, chunk_overlap=50)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_chunks_have_overlap() -> None:
    document = load_document(KB_DIR / "faq.md")

    chunks = chunk_document(document, chunk_size=180, chunk_overlap=40)

    assert len(chunks) > 1

    previous_words = set(chunks[0].content.split()[-8:])
    next_words = set(chunks[1].content.split()[:8])

    assert previous_words.intersection(next_words)


def test_chunk_documents_preserves_document_order() -> None:
    documents = [
        load_document(KB_DIR / "faq.md"),
        load_document(KB_DIR / "refund_policy.md"),
    ]

    chunks = chunk_documents(documents, chunk_size=250, chunk_overlap=50)

    assert chunks[0].document_id == "faq"
    assert any(chunk.document_id == "refund-policy" for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 10),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_invalid_chunk_configuration_is_rejected(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    document = load_document(KB_DIR / "faq.md")

    with pytest.raises(ChunkingError):
        chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )