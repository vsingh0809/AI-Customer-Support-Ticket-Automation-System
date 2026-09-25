from unittest.mock import Mock

import pytest

from app.ai.rag.generator import (
    GeneratedResponse,
    GenerationError,
    SourceReference,
)
from app.ai.rag.pipeline import RAGPipeline, RAGPipelineError
from app.ai.rag.retriever import RetrievalError
from app.ai.rag.vector_store import VectorSearchResult


def _result(
    source: str = "refund_policy.md",
    score: float = 0.92,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id="refund-policy-chunk-000",
        document_id="refund-policy",
        content="Customers may request a refund within 7 days.",
        score=score,
        metadata={
            "title": "Refund Policy",
            "category": "refund",
            "version": "1.0",
            "source": source,
            "chunk_index": 0,
            "total_chunks": 2,
        },
    )


def _response() -> GeneratedResponse:
    return GeneratedResponse(
        answer="Refunds are available within 7 days.",
        sources=(
            SourceReference(
                title="Refund Policy",
                source="refund_policy.md",
                category="refund",
                score=0.92,
                chunk_index=0,
            ),
        ),
    )


def test_rag_pipeline_success() -> None:
    retriever = Mock()
    generator = Mock()

    retriever.retrieve.return_value = [_result()]
    generator.generate.return_value = _response()

    pipeline = RAGPipeline(
        retriever= retriever,
        generator=generator,
    )

    result = pipeline.ask("What is the refund policy?")

    assert result.answer == "Refunds are available within 7 days."
    assert len(result.sources) == 1
    assert result.sources[0].source == "refund_policy.md"

    retriever.retrieve.assert_called_once_with(
        "What is the refund policy?"
    )

    generator.generate.assert_called_once()


def test_rag_pipeline_handles_no_relevant_results() -> None:
    retriever = Mock()
    generator = Mock()

    retriever.retrieve.return_value = []

    generator.generate.return_value = GeneratedResponse(
        answer=(
            "I couldn't find reliable information in the "
            "support knowledge base to answer that question."
        ),
        sources=(),
    )

    pipeline = RAGPipeline(
        retriever= retriever,
        generator=generator,
    )

    result = pipeline.ask(
        "What is the company's astronaut policy?"
    )

    assert result.sources == ()
    assert "couldn't find reliable information" in result.answer


def test_rag_pipeline_handles_retrieval_failure() -> None:
    retriever = Mock()
    generator = Mock()

    retriever.retrieve.side_effect = RetrievalError(
        "Qdrant unavailable"
    )

    pipeline = RAGPipeline(
        retriever= retriever,
        generator=generator,
    )

    with pytest.raises(
        RAGPipelineError,
        match="Knowledge retrieval failed",
    ):
        pipeline.ask("What is the refund policy?")

    generator.generate.assert_not_called()


def test_rag_pipeline_handles_generation_failure() -> None:
    retriever = Mock()
    generator = Mock()

    retriever.retrieve.return_value = [_result()]

    generator.generate.side_effect = GenerationError(
        "LLM unavailable"
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    with pytest.raises(
        RAGPipelineError,
        match="Grounded response generation failed",
    ):
        pipeline.ask("What is the refund policy?")


def test_rag_pipeline_rejects_empty_query() -> None:
    retriever = Mock()
    generator = Mock()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    with pytest.raises(
        RAGPipelineError,
        match="Query cannot be empty",
    ):
        pipeline.ask("   ")

    retriever.retrieve.assert_not_called()
    generator.generate.assert_not_called()