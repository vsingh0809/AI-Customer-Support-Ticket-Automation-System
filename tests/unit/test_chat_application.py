from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.services.chat_application import (
    ChatApplicationError,
    ChatApplicationService,
)


def _build_conversation(customer_id):
    conversation = Mock()
    conversation.id = uuid4()
    conversation.customer_id = customer_id
    return conversation


def _build_service(
    *,
    graph_result=None,
    history=None,
):
    memory_service = Mock()
    agent_graph = Mock()

    customer_id = uuid4()
    conversation = _build_conversation(customer_id)

    memory_service.create_conversation.return_value = conversation
    memory_service.get_conversation.return_value = conversation
    memory_service.load_history.return_value = history or []

    agent_graph.invoke.return_value = graph_result or {
        "response": "Your order is currently shipped."
    }

    service = ChatApplicationService(
        memory_service=memory_service,
        agent_graph=agent_graph,
    )

    return (
        service,
        memory_service,
        agent_graph,
        customer_id,
        conversation,
    )


def test_creates_conversation_for_new_chat() -> None:
    (
        service,
        memory_service,
        agent_graph,
        customer_id,
        conversation,
    ) = _build_service()

    conversation_id, response = service.chat(
        customer_id=customer_id,
        message="Where is my order 45821?",
    )

    assert conversation_id == conversation.id
    assert response == "Your order is currently shipped."

    memory_service.create_conversation.assert_called_once_with(
        customer_id=customer_id,
    )

    memory_service.get_conversation.assert_not_called()
    agent_graph.invoke.assert_called_once()


def test_loads_existing_conversation() -> None:
    (
        service,
        memory_service,
        agent_graph,
        customer_id,
        conversation,
    ) = _build_service(
        history=[
            {
                "role": "user",
                "content": "My order is 45821.",
            },
            {
                "role": "assistant",
                "content": "I can help with that.",
            },
        ]
    )

    service.chat(
        customer_id=customer_id,
        message="When will it arrive?",
        conversation_id=conversation.id,
    )

    memory_service.get_conversation.assert_called_once_with(
        conversation_id=conversation.id,
        customer_id=customer_id,
    )

    agent_state = agent_graph.invoke.call_args.args[0]

    assert agent_state["conversation_id"] == conversation.id
    assert agent_state["customer_id"] == customer_id
    assert agent_state["user_message"] == "When will it arrive?"
    assert agent_state["conversation_history"] == [
        {
            "role": "user",
            "content": "My order is 45821.",
        },
        {
            "role": "assistant",
            "content": "I can help with that.",
        },
    ]


def test_persists_user_and_assistant_turns() -> None:
    (
        service,
        memory_service,
        _,
        customer_id,
        conversation,
    ) = _build_service()

    service.chat(
        customer_id=customer_id,
        message="  Where is my order 45821?  ",
    )

    memory_service.append_turns.assert_called_once_with(
        conversation_id=conversation.id,
        customer_id=customer_id,
        turns=[
            {
                "role": "user",
                "content": "Where is my order 45821?",
            },
            {
                "role": "assistant",
                "content": "Your order is currently shipped.",
            },
        ],
    )


def test_rejects_missing_conversation() -> None:
    service, memory_service, _, customer_id, conversation = _build_service()

    memory_service.get_conversation.return_value = None

    with pytest.raises(
        ChatApplicationError,
        match="Conversation not found",
    ):
        service.chat(
            customer_id=customer_id,
            message="Hello",
            conversation_id=conversation.id,
        )


def test_rejects_empty_message() -> None:
    service, _, _, customer_id, _ = _build_service()

    with pytest.raises(
        ChatApplicationError,
        match="Customer message cannot be empty",
    ):
        service.chat(
            customer_id=customer_id,
            message="   ",
        )


def test_rejects_empty_agent_response() -> None:
    (
        service,
        _,
        agent_graph,
        customer_id,
        _,
    ) = _build_service(
        graph_result={"response": ""},
    )

    with pytest.raises(
        ChatApplicationError,
        match="workflow returned no response",
    ):
        service.chat(
            customer_id=customer_id,
            message="Hello",
        )

    agent_graph.invoke.assert_called_once()