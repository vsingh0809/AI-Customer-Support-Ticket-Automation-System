"""Knowledge-base ingestion pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from app.ai.rag.chunker import chunk_documents
from app.ai.rag.embeddings import EmbeddingService, FastEmbedProvider
from app.ai.rag.loader import load_documents
from app.ai.rag.vector_store import QdrantVectorStore, VectorStoreError
from app.core.config import get_settings


class KnowledgeBaseIngestionError(RuntimeError):
    """Raised when knowledge-base ingestion fails."""


class KnowledgeBaseIngestionService:
    """Load, chunk, embed, and index knowledge-base documents."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def ingest(self, directory: Path | str) -> int:
        """Ingest all manifest-controlled KB documents into Qdrant."""
        kb_directory = Path(directory)

        try:
            documents = load_documents(kb_directory)
            chunks = chunk_documents(documents)

            if not chunks:
                raise KnowledgeBaseIngestionError(
                    "Knowledge base produced no chunks."
                )

            embedded_chunks = self._embedding_service.embed_chunks(chunks)

            return self._vector_store.upsert_chunks(embedded_chunks)

        except KnowledgeBaseIngestionError:
            raise
        except VectorStoreError as exc:
            raise KnowledgeBaseIngestionError(
                "Failed to index knowledge base."
            ) from exc
        except Exception as exc:
            raise KnowledgeBaseIngestionError(
                "Knowledge-base ingestion failed."
            ) from exc


def build_ingestion_service() -> KnowledgeBaseIngestionService:
    """Build the production ingestion dependencies."""
    settings = get_settings()

    if not settings.qdrant_url:
        raise KnowledgeBaseIngestionError(
            "QDRANT_URL is required for knowledge-base ingestion."
        )

    embedding_provider = FastEmbedProvider(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    embedding_service = EmbeddingService(
        embedding_provider,
        batch_size=settings.embedding_batch_size,
    )

    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )

    vector_store = QdrantVectorStore(
        qdrant_client,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.qdrant_vector_size,
    )

    return KnowledgeBaseIngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def main() -> None:
    """Run the knowledge-base ingestion CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest Markdown knowledge-base documents into Qdrant."
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("knowledge_base"),
        help="Path to the knowledge-base directory.",
    )

    args = parser.parse_args()

    service = build_ingestion_service()
    count = service.ingest(args.directory)

    print(f"Knowledge-base ingestion complete. Indexed {count} chunks.")


if __name__ == "__main__":
    main()