from unittest.mock import Mock
from uuid import uuid4

from app.db.models import Conversation, Message
from app.db.repositories import (
    ConversationRepository,
    MessageRepository,
)


def test_conversation_repository_get_by_id() -> None:
    db = Mock()

    conversation = Conversation(
        id=uuid4(),
        customer_id=uuid4(),
    )

    db.scalar.return_value = conversation

    repository = ConversationRepository(db)

    result = repository.get_by_id(
        conversation.id,
        conversation.customer_id,
    )

    assert result is conversation
    db.scalar.assert_called_once()

def test_conversation_repository_add_flushes() -> None:
        db = Mock()
        repository = ConversationRepository(db)

        conversation = Conversation(
            id=uuid4(),
            customer_id=uuid4(),
        )

        result = repository.add(conversation)

        assert result is conversation
        db.add.assert_called_once_with(conversation)
        db.flush.assert_called_once()

def test_message_repository_add_flushes() -> None:
    db = Mock()
    repository = MessageRepository(db)

    message = Message(
        id=uuid4(),
        conversation_id=uuid4(),
        role="user",
        content="Hello",
    )

    result = repository.add(message)

    assert result is message
    db.add.assert_called_once_with(message)
    db.flush.assert_called_once()