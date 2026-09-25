"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.ai.rag.generator import (
    GeneratedResponse,
    GenerationError,
    GroundedResponseGenerator,
)
from app.ai.rag.retriever import RetrievalError, Retriever


class RAGPipelineError(RuntimeError):
    """Raised when the RAG pipeline fails."""


class RAGResponse(BaseModel):
    """Final response returned by the RAG pipeline."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    sources: tuple = ()


class RAGPipeline:
    """Coordinate retrieval and grounded generation."""

    def __init__(
        self,
        retriever: Retriever,
        generator: GroundedResponseGenerator,
    ) -> None:
        self._retriever = retriever
        self._generator = generator

    def ask(self, query: str) -> RAGResponse:
        """Retrieve evidence and generate a grounded response."""
        if not query.strip():
            raise RAGPipelineError("Query cannot be empty")

        try:
            results = self._retriever.retrieve(query)
        except RetrievalError as exc:
            raise RAGPipelineError(
                "Knowledge retrieval failed"
            ) from exc

        try:
            response: GeneratedResponse = self._generator.generate(
                query=query,
                results=results,
            )
        except GenerationError as exc:
            raise RAGPipelineError(
                "Grounded response generation failed"
            ) from exc

        return RAGResponse(
            answer=response.answer,
            sources=response.sources,
        )