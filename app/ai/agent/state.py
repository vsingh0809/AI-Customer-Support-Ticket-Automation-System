"""State contract shared across agent orchestration nodes."""

from __future__ import annotations

from typing import Any, Literal, TypedDict
from uuid import UUID

from app.ai.agent.contracts import AgentRoute, Intent
from app.ai.rag.generator import SourceReference
from app.ai.rag.vector_store import VectorSearchResult


class ConversationTurn(TypedDict):
    """A single turn in the customer conversation."""

    role: Literal["user", "assistant"]
    content: str


class AgentState(TypedDict, total=False):
    """State carried through the LangGraph workflow."""

    conversation_id: UUID
    customer_id: UUID

    conversation_history: list[ConversationTurn]

    user_message: str

    intent: Intent | None
    route: AgentRoute | None

    entities: dict[str, Any]

    retrieved_context: list[VectorSearchResult]
    sources: list[SourceReference]

    tool_name: str | None
    tool_arguments: dict[str, Any]
    tool_result: dict[str, Any] | None
    missing_fields: list[str]
    clarification_question: str | None
    response: str | None

    ticket_id: str | None
    escalated: bool

    errors: list[str]