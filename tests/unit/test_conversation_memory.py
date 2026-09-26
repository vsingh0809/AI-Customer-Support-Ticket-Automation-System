from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from app.services.conversation_memory import ConversationMemoryService


def test_load_history_returns_persisted_conversation_turns() -> None:
    conversation_id = uuid4()
    customer_id = uuid4()

    conversation_repository = Mock()
    message_repository = Mock()

    conversation_repository.get_by_id.return_value = SimpleNamespace(
        id=conversation_id,
        customer_id=customer_id,
    )

    message_repository.list_by_conversation.return_value = [
        SimpleNamespace(
            role="user",
            content="Where is my order 45821?",
        ),
        SimpleNamespace(
            role="assistant",
            content="Your order is shipped.",
        ),
    ]

    service = ConversationMemoryService(
        conversation_repository,
        message_repository,
    )

    result = service.load_history(
        conversation_id,
        customer_id,
    )

    assert result == [
        {
            "role": "user",
            "content": "Where is my order 45821?",
        },
        {
            "role": "assistant",
            "content": "Your order is shipped.",
        },
    ]

    conversation_repository.get_by_id.assert_called_once_with(
        conversation_id=conversation_id,
        customer_id=customer_id,
    )

    message_repository.list_by_conversation.assert_called_once_with(
        conversation_id,
    )


def test_load_history_returns_empty_for_missing_conversation() -> None:
    conversation_repository = Mock()
    message_repository = Mock()

    conversation_repository.get_by_id.return_value = None

    service = ConversationMemoryService(
        conversation_repository,
        message_repository,
    )

    result = service.load_history(
        uuid4(),
        uuid4(),
    )

    assert result == []

    message_repository.list_by_conversation.assert_not_called()


def test_append_turns_persists_messages() -> None:
    conversation_id = uuid4()
    customer_id = uuid4()

    conversation = SimpleNamespace(
        id=conversation_id,
        customer_id=customer_id,
        updated_at=None,
    )

    conversation_repository = Mock()
    message_repository = Mock()

    conversation_repository.get_by_id.return_value = conversation

    service = ConversationMemoryService(
        conversation_repository,
        message_repository,
    )

    turns = [
        {
            "role": "user",
            "content": "Where is order 45821?",
        },
        {
            "role": "assistant",
            "content": "Your order is shipped.",
        },
    ]

    service.append_turns(
        conversation_id,
        customer_id,
        turns,
    )

    assert message_repository.add.call_count == 2

    first_message = message_repository.add.call_args_list[0].args[0]
    second_message = message_repository.add.call_args_list[1].args[0]

    assert first_message.conversation_id == conversation_id
    assert first_message.role == "user"
    assert first_message.content == "Where is order 45821?"

    assert second_message.conversation_id == conversation_id
    assert second_message.role == "assistant"
    assert second_message.content == "Your order is shipped."

    assert conversation.updated_at is not None


def test_append_turns_rejects_missing_conversation() -> None:
    conversation_repository = Mock()
    message_repository = Mock()

    conversation_repository.get_by_id.return_value = None

    service = ConversationMemoryService(
        conversation_repository,
        message_repository,
    )

    try:
        service.append_turns(
            uuid4(),
            uuid4(),
            [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
        )
    except ValueError as exc:
        assert str(exc) == "Conversation not found"
    else:
        raise AssertionError(
            "Expected ValueError"
        )

    message_repository.add.assert_not_called()