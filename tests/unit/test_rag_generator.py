from unittest.mock import Mock

import pytest

from app.ai.rag.generator import (
    DeepSeekProvider,
    GeneratedResponse,
    GenerationError,
    GroundedResponseGenerator,
)
from app.ai.rag.vector_store import VectorSearchResult


class FakeLLMProvider:
    def __init__(self, answer: str = "Refunds are available within 7 days.") -> None:
        self.answer = answer
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.answer


def _result(
    content: str = "Customers may request a refund within 7 days.",
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id="refund-policy-chunk-000",
        document_id="refund-policy",
        content=content,
        score=0.92,
        metadata={
            "title": "Refund Policy",
            "category": "refund",
            "version": "1.0",
            "source": "refund_policy.md",
            "chunk_index": 0,
            "total_chunks": 2,
        },
    )


def test_generator_returns_grounded_response() -> None:
    provider = FakeLLMProvider()

    generator = GroundedResponseGenerator(provider)

    result = generator.generate(
        query="Can I get a refund?",
        results=[_result()],
    )

    assert isinstance(result, GeneratedResponse)
    assert result.answer == "Refunds are available within 7 days."


def test_generator_includes_retrieved_context() -> None:
    provider = FakeLLMProvider()

    generator = GroundedResponseGenerator(provider)

    generator.generate(
        query="What is the refund policy?",
        results=[_result()],
    )

    assert "Customers may request a refund within 7 days." in provider.user_prompt
    assert "refund_policy.md" in provider.user_prompt


def test_generator_includes_grounding_rules() -> None:
    provider = FakeLLMProvider()

    generator = GroundedResponseGenerator(provider)

    generator.generate(
        query="What is the refund policy?",
        results=[_result()],
    )
    normalized_prompt = " ".join(provider.system_prompt.split())

    assert "ONLY the supplied knowledge-base context" in normalized_prompt
    assert "Do not invent company policies" in normalized_prompt



def test_generator_returns_safe_fallback_without_context() -> None:
    provider = Mock()

    generator = GroundedResponseGenerator(provider)

    result = generator.generate(
        query="What is your refund policy?",
        results=[],
    )

    assert result.answer == (
        "I couldn't find reliable information in the support "
        "knowledge base to answer that question."
    )

    provider.generate.assert_not_called()


def test_generator_rejects_empty_query() -> None:
    provider = Mock()

    generator = GroundedResponseGenerator(provider)

    with pytest.raises(
        GenerationError,
        match="Query cannot be empty",
    ):
        generator.generate(
            query="   ",
            results=[_result()],
        )


def test_deepseek_provider_rejects_empty_api_key() -> None:
    with pytest.raises(
        GenerationError,
        match="API key cannot be empty",
    ):
        DeepSeekProvider(api_key="   ")


def test_deepseek_provider_rejects_empty_prompts() -> None:
    provider = Mock()

    deepseek = DeepSeekProvider(
        api_key="test-key",
        client=provider,
    )

    with pytest.raises(
        GenerationError,
        match="System prompt cannot be empty",
    ):
        deepseek.generate(
            system_prompt=" ",
            user_prompt="hello",
        )

    with pytest.raises(
        GenerationError,
        match="User prompt cannot be empty",
    ):
        deepseek.generate(
            system_prompt="system",
            user_prompt=" ",
        )


def test_deepseek_provider_maps_response() -> None:
    response = Mock()
    response.choices = [
        Mock(
            message=Mock(
                content="Refunds are available within 7 days."
            )
        )
    ]

    client = Mock()
    client.chat.completions.create.return_value = response

    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-flash",
        client=client,
    )

    result = provider.generate(
        system_prompt="You are a support assistant.",
        user_prompt="What is the refund policy?",
    )

    assert result == "Refunds are available within 7 days."

    client.chat.completions.create.assert_called_once()

def test_generator_returns_source_references() -> None:
    provider = FakeLLMProvider()

    generator = GroundedResponseGenerator(provider)

    result = generator.generate(
        query="What is the refund policy?",
        results=[_result()],
    )

    assert len(result.sources) == 1

    source = result.sources[0]

    assert source.title == "Refund Policy"
    assert source.source == "refund_policy.md"
    assert source.category == "refund"
    assert source.score == 0.92
    assert source.chunk_index == 0    

def test_generator_deduplicates_sources() -> None:
    provider = FakeLLMProvider()

    first = _result(
        content="Refunds are available within 7 days."
    )

    second = VectorSearchResult(
        chunk_id="refund-policy-chunk-001",
        document_id="refund-policy",
        content="Refund requests must meet eligibility requirements.",
        score=0.80,
        metadata={
            "title": "Refund Policy",
            "category": "refund",
            "version": "1.0",
            "source": "refund_policy.md",
            "chunk_index": 1,
            "total_chunks": 2,
        },
    )

    generator = GroundedResponseGenerator(provider)

    result = generator.generate(
        query="What is the refund policy?",
        results=[first, second],
    )

    assert len(result.sources) == 1
    assert result.sources[0].source == "refund_policy.md"
    assert result.sources[0].score == 0.92

def test_generator_keeps_distinct_sources() -> None:
    provider = FakeLLMProvider()

    first = _result()

    second = VectorSearchResult(
        chunk_id="shipping-policy-chunk-001",
        document_id="shipping-policy",
        content="Standard delivery takes 3 to 5 business days.",
        score=0.88,
        metadata={
            "title": "Shipping Policy",
            "category": "shipping",
            "version": "1.0",
            "source": "shipping_policy.md",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    )

    generator = GroundedResponseGenerator(provider)

    result = generator.generate(
        query="Tell me about refunds and shipping.",
        results=[first, second],
    )

    assert len(result.sources) == 2

    assert result.sources[0].source == "refund_policy.md"
    assert result.sources[1].source == "shipping_policy.md"    

def test_generator_returns_no_sources_without_context() -> None:
    provider = Mock()

    generator = GroundedResponseGenerator(provider)

    result = generator.generate(
        query="Unknown question",
        results=[],
    )

    assert result.sources == ()