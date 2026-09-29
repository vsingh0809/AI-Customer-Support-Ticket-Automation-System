from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

from app.db.models import Conversation, Message
from app.services.conversation_memory import (
    ConversationMemoryService,
)


def _build_service():
    conversation_repository = Mock()
    message_repository = Mock()

    service = ConversationMemoryService(
        conversation_repository=conversation_repository,
        message_repository=message_repository,
    )

    return (
        service,
        conversation_repository,
        message_repository,
    )


def test_list_conversations_returns_sidebar_summaries() -> None:
    (
        service,
        conversation_repository,
        message_repository,
    ) = _build_service()

    customer_id = uuid4()
    conversation_id = uuid4()
    updated_at = datetime.now(UTC)

    conversation = Conversation(
        id=conversation_id,
        customer_id=customer_id,
        status="active",
        created_at=updated_at,
        updated_at=updated_at,
    )

    first_message = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        role="user",
        content="Where is my order 45821?",
        created_at=updated_at,
    )

    conversation_repository.list_by_customer.return_value = [
        conversation
    ]

    message_repository.get_first_user_message.return_value = (
        first_message
    )

    result = service.list_conversations(customer_id)

    assert len(result) == 1
    assert result[0].conversation_id == conversation_id
    assert result[0].title == "Where is my order 45821?"
    assert result[0].updated_at == updated_at


def test_conversation_title_is_truncated() -> None:
    (
        service,
        _conversation_repository,
        _message_repository,
    ) = _build_service()

    long_message = "A" * 100

    result = service._build_title(long_message)

    assert len(result) == 60
    assert result.endswith("...")


def test_conversation_title_uses_new_conversation_for_empty_message() -> None:
    (
        service,
        _conversation_repository,
        _message_repository,
    ) = _build_service()

    result = service._build_title(None)

    assert result == "New conversation"


def test_load_conversation_messages_checks_customer_ownership() -> None:
    (
        service,
        conversation_repository,
        message_repository,
    ) = _build_service()

    customer_id = uuid4()
    conversation_id = uuid4()

    conversation = Conversation(
        id=conversation_id,
        customer_id=customer_id,
        status="active",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    user_message = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        role="user",
        content="Where is my order?",
        created_at=datetime.now(UTC),
    )

    assistant_message = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        role="assistant",
        content="Please provide your order ID.",
        created_at=datetime.now(UTC),
    )

    conversation_repository.get_by_id.return_value = conversation
    message_repository.list_by_conversation.return_value = [
        user_message,
        assistant_message,
    ]

    result = service.load_conversation_messages(
        conversation_id=conversation_id,
        customer_id=customer_id,
    )

    assert result is not None
    assert len(result) == 2

    assert result[0].role == "user"
    assert result[0].content == "Where is my order?"

    assert result[1].role == "assistant"
    assert result[1].content == (
        "Please provide your order ID."
    )


def test_load_conversation_messages_returns_none_for_unknown_conversation() -> None:
    (
        service,
        conversation_repository,
        message_repository,
    ) = _build_service()

    conversation_repository.get_by_id.return_value = None

    result = service.load_conversation_messages(
        conversation_id=uuid4(),
        customer_id=uuid4(),
    )

    assert result is None
    message_repository.list_by_conversation.assert_not_called()