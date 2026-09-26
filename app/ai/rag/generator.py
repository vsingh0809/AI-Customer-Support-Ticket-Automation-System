"""Grounded response generation for retrieved knowledge."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.ai.rag.vector_store import VectorSearchResult


class GenerationError(RuntimeError):
    """Raised when grounded response generation fails."""


class LLMProvider(Protocol):
    """Provider contract for response generation."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response from the supplied prompts."""


class SourceReference(BaseModel):
    """A source supporting a generated answer."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    category: str = Field(min_length=1)
    score: float
    chunk_index: int = Field(ge=0)


class GeneratedResponse(BaseModel):
    """Grounded answer together with supporting sources."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    sources: tuple[SourceReference, ...] = ()


class DeepSeekProvider:
    """DeepSeek chat provider using its OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-flash",
        temperature: float = 0.0,
        max_tokens: int = 800,
        client: OpenAI | None = None,
    ) -> None:
        if not api_key.strip():
            raise GenerationError("DeepSeek API key cannot be empty")

        if not model.strip():
            raise GenerationError("LLM model cannot be empty")

        if temperature < 0:
            raise GenerationError("Temperature cannot be negative")

        if max_tokens <= 0:
            raise GenerationError(
                "max_tokens must be greater than zero"
            )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._client = client or OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response from DeepSeek."""
        if not system_prompt.strip():
            raise GenerationError("System prompt cannot be empty")

        if not user_prompt.strip():
            raise GenerationError("User prompt cannot be empty")

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            raise GenerationError(
                "DeepSeek generation request failed"
            ) from exc

        if not response.choices:
            raise GenerationError(
                "DeepSeek returned no completion choices"
            )

        content = response.choices[0].message.content

        if not content or not content.strip():
            raise GenerationError(
                "DeepSeek returned an empty response"
            )

        return content.strip()

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        """Generate and parse a JSON response from DeepSeek."""
        if not system_prompt.strip():
            raise GenerationError("System prompt cannot be empty")

        if not user_prompt.strip():
            raise GenerationError("User prompt cannot be empty")

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise GenerationError(
                "DeepSeek JSON generation request failed"
            ) from exc

        if not response.choices:
            raise GenerationError(
                "DeepSeek returned no completion choices"
            )

        content = response.choices[0].message.content

        if not content or not content.strip():
            raise GenerationError(
                "DeepSeek returned empty JSON content"
            )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise GenerationError(
                "DeepSeek returned invalid JSON"
            ) from exc

        if not isinstance(parsed, dict):
            raise GenerationError(
                "DeepSeek JSON response must be an object"
            )

        return parsed


class GroundedResponseGenerator:
    """Generate answers constrained by retrieved knowledge."""

    def __init__(
        self,
        provider: LLMProvider,
    ) -> None:
        self._provider = provider

    def generate(
    self,
    *,
    query: str,
    results: Sequence[VectorSearchResult],
    ) -> GeneratedResponse:
        """Generate a grounded answer and expose supporting sources."""
        if not query.strip():
            raise GenerationError("Query cannot be empty")

        if not results:
            return GeneratedResponse(
                answer=(
                    "I couldn't find reliable information in the "
                    "support knowledge base to answer that question."
                ),
                sources=(),
            )

        context = _build_context(results)

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(
            query=query,
            context=context,
        )

        answer = self._provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        sources = _build_sources(results)

        return GeneratedResponse(
            answer=answer,
            sources=tuple(sources),
        )


def _build_system_prompt() -> str:
    """Build the grounding rules for the model."""
    return """You are NovaMart's customer support assistant.

Answer the customer's question using ONLY the supplied knowledge-base
context.

Rules:
1. Do not invent company policies, prices, timelines, eligibility rules,
   procedures, or other company-specific facts.
2. Do not use outside knowledge to fill gaps.
3. If the context does not contain enough information, explicitly say that
   the knowledge base does not provide enough information.
4. Keep the answer concise and directly relevant to the customer.
5. Never claim that an action was completed unless an application tool
   actually completed that action.
"""


def _build_user_prompt(
    *,
    query: str,
    context: str,
) -> str:
    """Build the user prompt containing retrieved evidence."""
    return f"""Knowledge-base context:

{context}

Customer question:

{query}

Answer the customer using only the knowledge-base context above.
"""


def _build_context(
    results: Sequence[VectorSearchResult],
) -> str:
    """Format retrieved chunks for the grounded prompt."""
    sections: list[str] = []

    for index, result in enumerate(results, start=1):
        sections.append(
            "\n".join(
                [
                    f"[Context {index}]",
                    f"Source: {result.metadata['source']}",
                    f"Category: {result.metadata['category']}",
                    f"Content: {result.content}",
                ]
            )
        )

    return "\n\n".join(sections)

def _build_sources(
    results: Sequence[VectorSearchResult],
) -> list[SourceReference]:
    """Build unique source references, keeping the strongest chunk."""
    sources_by_source: dict[str, SourceReference] = {}

    for result in results:
        source = str(result.metadata["source"])

        reference = SourceReference(
            title=str(result.metadata["title"]),
            source=source,
            category=str(result.metadata["category"]),
            score=result.score,
            chunk_index=int(result.metadata["chunk_index"]),
        )

        existing = sources_by_source.get(source)

        if existing is None or reference.score > existing.score:
            sources_by_source[source] = reference

    return list(sources_by_source.values())