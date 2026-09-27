"""Application service for customer chat requests."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.ai.agent.state import AgentState
from app.services.conversation_memory import ConversationMemoryService


class CompiledAgentGraph(Protocol):
    """Contract for the compiled LangGraph used by the application."""

    def invoke(self, state: AgentState) -> AgentState:
        """Execute one customer-support workflow."""


class ChatApplicationError(RuntimeError):
    """Raised when the chat application workflow fails."""


class ChatApplicationService:
    """Orchestrate conversation persistence and agent execution."""

    def __init__(
        self,
        *,
        memory_service: ConversationMemoryService,
        agent_graph: CompiledAgentGraph,
    ) -> None:
        self._memory_service = memory_service
        self._agent_graph = agent_graph

    def chat(
        self,
        *,
        customer_id: UUID,
        message: str,
        conversation_id: UUID | None = None,
    ) -> tuple[UUID, str]:
        """Process one customer message."""
        cleaned_message = message.strip()

        if not cleaned_message:
            raise ChatApplicationError(
                "Customer message cannot be empty"
            )

        conversation = self._get_or_create_conversation(
            customer_id=customer_id,
            conversation_id=conversation_id,
        )

        history = self._memory_service.load_history(
            conversation_id=conversation.id,
            customer_id=customer_id,
        )

        state: AgentState = {
            "conversation_id": conversation.id,
            "customer_id": customer_id,
            "user_message": cleaned_message,
            "conversation_history": history,
            "errors": [],
        }

        try:
            result = self._agent_graph.invoke(state)
        except Exception as exc:
            raise ChatApplicationError(
                "Customer-support workflow failed"
            ) from exc

        response = result.get("response")

        if not isinstance(response, str) or not response.strip():
            raise ChatApplicationError(
                "Customer-support workflow returned no response"
            )

        new_turns = [
            {
                "role": "user",
                "content": cleaned_message,
            },
            {
                "role": "assistant",
                "content": response.strip(),
            },
        ]

        try:
            self._memory_service.append_turns(
                conversation_id=conversation.id,
                customer_id=customer_id,
                turns=new_turns,
            )
        except Exception as exc:
            raise ChatApplicationError(
                "Failed to persist conversation"
            ) from exc

        return conversation.id, response.strip()

    def _get_or_create_conversation(
        self,
        *,
        customer_id: UUID,
        conversation_id: UUID | None,
    ):
        """Load an owned conversation or create a new one."""
        if conversation_id is None:
            return self._memory_service.create_conversation(
                customer_id=customer_id,
            )

        conversation = self._memory_service.get_conversation(
            conversation_id=conversation_id,
            customer_id=customer_id,
        )

        if conversation is None:
            raise ChatApplicationError(
                "Conversation not found"
            )

        return conversation