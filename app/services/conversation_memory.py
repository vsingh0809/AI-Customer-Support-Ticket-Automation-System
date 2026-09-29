from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.ai.agent.state import ConversationTurn
from app.db.models import Conversation, Message
from app.db.repositories import (
    ConversationRepository,
    MessageRepository,
)


@dataclass(frozen=True)
class ConversationSummary:
    """Summary displayed in the customer conversation sidebar."""

    conversation_id: UUID
    title: str
    updated_at: datetime


@dataclass(frozen=True)
class ConversationMessage:
    """Persisted customer-support conversation message."""

    role: str
    content: str
    created_at: datetime


class ConversationMemoryService:
    """Persist and load customer conversation history."""

    _TITLE_MAX_LENGTH = 60

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
        now = datetime.now(UTC)

        conversation = Conversation(
            id=uuid4(),
            customer_id=customer_id,
            status="active",
            created_at=now,
            updated_at=now,
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

    def list_conversations(
        self,
        customer_id: UUID,
    ) -> list[ConversationSummary]:
        """Return customer conversations for sidebar display."""

        conversations = (
            self._conversation_repository.list_by_customer(
                customer_id
            )
        )

        summaries: list[ConversationSummary] = []

        for conversation in conversations:
            first_message = (
                self._message_repository.get_first_user_message(
                    conversation.id
                )
            )

            title = self._build_title(
                first_message.content
                if first_message is not None
                else None
            )

            summaries.append(
                ConversationSummary(
                    conversation_id=conversation.id,
                    title=title,
                    updated_at=conversation.updated_at,
                )
            )

        return summaries

    def load_conversation_messages(
        self,
        conversation_id: UUID,
        customer_id: UUID,
    ) -> list[ConversationMessage] | None:
        """Load persisted messages for an owned conversation."""

        conversation = self.get_conversation(
            conversation_id=conversation_id,
            customer_id=customer_id,
        )

        if conversation is None:
            return None

        messages = self._message_repository.list_by_conversation(
            conversation.id,
        )

        return [
            ConversationMessage(
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in messages
            if message.role in {"user", "assistant"}
        ]

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

        now = datetime.now(UTC)

        for turn in turns:
            self._message_repository.add(
                Message(
                    id=uuid4(),
                    conversation_id=conversation.id,
                    role=turn["role"],
                    content=turn["content"],
                    created_at=now,
                )
            )

        conversation.updated_at = datetime.now(UTC)

    @classmethod
    def _build_title(cls, content: str | None) -> str:
        """Build a compact sidebar title from the first customer message."""

        if not content:
            return "New conversation"

        cleaned = " ".join(content.split())

        if len(cleaned) <= cls._TITLE_MAX_LENGTH:
            return cleaned

        return (
            cleaned[: cls._TITLE_MAX_LENGTH - 3].rstrip()
            + "..."
        )