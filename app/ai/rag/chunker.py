"""Split normalized knowledge-base documents into retrieval-ready chunks."""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, ConfigDict, Field

from app.ai.rag.loader import KnowledgeDocument


class KnowledgeChunk(BaseModel):
    """A retrieval-ready chunk derived from a knowledge-base document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(gt=0)
    content: str = Field(min_length=1)


class ChunkingError(ValueError):
    """Raised when chunking configuration or input is invalid."""


def chunk_document(
    document: KnowledgeDocument,
    *,
    chunk_size: int = 3200,
    chunk_overlap: int = 400,
) -> list[KnowledgeChunk]:
    """
    Split one document into deterministic overlapping chunks.

    chunk_size and chunk_overlap are character-based approximations:
    ~3200 characters is roughly 700–800 tokens for typical English text.
    """
    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ChunkingError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ChunkingError("chunk_overlap must be smaller than chunk_size")

    content = _normalize_chunk_whitespace(document.content)

    if not content:
        raise ChunkingError(
            f"Cannot chunk empty document: {document.metadata.document_id}"
        )

    if len(content) <= chunk_size:
        raw_chunks = [content]
    else:
        raw_chunks = _split_with_overlap(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    total_chunks = len(raw_chunks)

    return [
        KnowledgeChunk(
            chunk_id=_build_chunk_id(document.metadata.document_id, index, chunk),
            document_id=document.metadata.document_id,
            title=document.metadata.title,
            category=document.metadata.category,
            version=document.metadata.version,
            source=document.metadata.source,
            chunk_index=index,
            total_chunks=total_chunks,
            content=chunk,
        )
        for index, chunk in enumerate(raw_chunks)
    ]


def chunk_documents(
    documents: list[KnowledgeDocument],
    *,
    chunk_size: int = 3200,
    chunk_overlap: int = 400,
) -> list[KnowledgeChunk]:
    """Chunk a collection of documents while preserving document order."""
    chunks: list[KnowledgeChunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks


def _split_with_overlap(
    content: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text into word-safe overlapping chunks."""
    words = content.split()

    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        additional_length = len(word) if not current_words else len(word) + 1

        if current_words and current_length + additional_length > chunk_size:
            chunk = " ".join(current_words).strip()

            if chunk:
                chunks.append(chunk)

            overlap_words: list[str] = []
            overlap_length = 0

            for previous_word in reversed(current_words):
                extra = len(previous_word) if not overlap_words else len(previous_word) + 1

                if overlap_length + extra > chunk_overlap:
                    break

                overlap_words.append(previous_word)
                overlap_length += extra

            current_words = list(reversed(overlap_words))
            current_length = len(" ".join(current_words))

        current_words.append(word)
        current_length += additional_length

    final_chunk = " ".join(current_words).strip()

    if final_chunk:
        chunks.append(final_chunk)

    return chunks


def _normalize_chunk_whitespace(content: str) -> str:
    """Collapse whitespace without changing the semantic text."""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _build_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    """Create a deterministic identifier for one chunk."""
    payload = f"{document_id}:{chunk_index}:{content}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{document_id}-chunk-{chunk_index:03d}-{digest}"