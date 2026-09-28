"""State contract shared across agent orchestration nodes."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict
from uuid import UUID

from app.ai.agent.contracts import AgentRoute, Intent
from app.ai.rag.generator import SourceReference
from app.ai.rag.vector_store import VectorSearchResult


class ConversationTurn(TypedDict):
    """A single message in the conversation."""

    role: Literal["user", "assistant"]
    content: str

class PlannedToolAction(TypedDict):
    """One validated business action planned for the current turn."""

    intent: Intent
    tool_name: str
    tool_arguments: dict[str, Any]
    tool_result: dict[str, Any] | None

def _add_and_cap_history(
    existing: list[ConversationTurn],
    new: list[ConversationTurn],
) -> list[ConversationTurn]:
    """Append new conversation turns and retain the latest 12 messages."""
    return (existing + new)[-12:]

class ActionType:
    """Supported workflow action types."""

    TOOL = "tool"
    RAG = "rag"



class PlannedAction(TypedDict):
    """One planned workflow action for the current customer request."""

    intent: Intent
    action_type: str
    tool_name: str | None

class PreparedAction(TypedDict):
    """One planned action with validated execution arguments."""

    intent: Intent
    action_type: str
    tool_name: str | None
    tool_arguments: dict[str, Any] | None


class ActionExecutionResult(TypedDict):
    """Result produced by executing one planned action."""

    intent: Intent
    action_type: str
    tool_name: str | None
    success: bool
    response: str | None
    tool_result: dict[str, Any] | None


class AgentState(TypedDict, total=False):
    """State carried through the LangGraph workflow."""

    conversation_id: UUID
    customer_id: UUID

    conversation_history: Annotated[
        list[ConversationTurn],
        _add_and_cap_history,
    ]

    user_message: str

    intent: Intent | None
    intents: list[Intent] | None
    planned_actions: list[PlannedAction]

    planned_tool_actions: list[PlannedToolAction]
    prepared_actions: list[PreparedAction]
    action_results: list[ActionExecutionResult]

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