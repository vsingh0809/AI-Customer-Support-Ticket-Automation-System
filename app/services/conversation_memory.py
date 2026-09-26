from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.ai.agent.state import ConversationTurn
from app.db.models import Conversation, Message
from app.db.repositories import (
    ConversationRepository,
    MessageRepository,
)


class ConversationMemoryService:
    """Persist and load customer conversation history."""

    def __init__(
        self,
        conversation_repository: ConversationRepository,
        message_repository: MessageRepository,
    ) -> None:
        self._conversation_repository = conversation_repository
        self._message_repository = message_repository

    def create_conversation(
        self,
        customer_id: UUID,
    ) -> Conversation:
        """Create a new active conversation."""
        conversation = Conversation(
            id=uuid4(),
            customer_id=customer_id,
            status="active",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        return self._conversation_repository.add(conversation)

    def get_conversation(
        self,
        conversation_id: UUID,
        customer_id: UUID,
    ) -> Conversation | None:
        """Load a conversation owned by the specified customer."""
        return self._conversation_repository.get_by_id(
            conversation_id=conversation_id,
            customer_id=customer_id,
        )

    def load_history(
        self,
        conversation_id: UUID,
        customer_id: UUID,
    ) -> list[ConversationTurn]:
        """Load all persisted conversation messages."""
        conversation = self.get_conversation(
            conversation_id=conversation_id,
            customer_id=customer_id,
        )

        if conversation is None:
            return []

        messages = self._message_repository.list_by_conversation(
            conversation.id,
        )

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
            if message.role in {"user", "assistant"}
        ]

    def append_turns(
        self,
        conversation_id: UUID,
        customer_id: UUID,
        turns: list[ConversationTurn],
    ) -> None:
        """Persist new conversation messages."""

        conversation = self.get_conversation(
            conversation_id=conversation_id,
            customer_id=customer_id,
        )

        if conversation is None:
            raise ValueError("Conversation not found")

        for turn in turns:
            self._message_repository.add(
                Message(
                    id=uuid4(),
                    conversation_id=conversation.id,
                    role=turn["role"],
                    content=turn["content"],
                    created_at=datetime.now(UTC),
                )
            )

        conversation.updated_at = datetime.now(UTC)