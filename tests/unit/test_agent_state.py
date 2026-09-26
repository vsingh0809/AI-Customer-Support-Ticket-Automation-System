from uuid import uuid4

from app.ai.agent.state import AgentState, _add_and_cap_history


def test_agent_state_supports_conversation_history() -> None:
    state: AgentState = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "conversation_history": [
            {
                "role": "user",
                "content": "Where is my order 45821?",
            },
            {
                "role": "assistant",
                "content": "Your order is currently shipped.",
            },
        ],
        "user_message": "When will it arrive?",
        "errors": [],
    }

    assert state["conversation_history"] == [
        {
            "role": "user",
            "content": "Where is my order 45821?",
        },
        {
            "role": "assistant",
            "content": "Your order is currently shipped.",
        },
    ]

    assert state["user_message"] == "When will it arrive?"


def test_conversation_history_reducer_caps_to_latest_12_messages() -> None:
        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"message-{index}",
            }
            for index in range(12)
        ]

        new_turns = [
            {
                "role": "user",
                "content": "message-12",
            },
            {
                "role": "assistant",
                "content": "message-13",
            },
        ]

        state: AgentState = {
            "conversation_id": uuid4(),
            "customer_id": uuid4(),
            "conversation_history": history,
            "user_message": "message-12",
            "errors": [],
        }

        result = _add_and_cap_history(
            state["conversation_history"],
            new_turns,
        )

        assert len(result) == 12
        assert result[0]["content"] == "message-2"
        assert result[-1]["content"] == "message-13"


def test_conversation_history_reducer_preserves_order() -> None:
    

    existing = [
        {
            "role": "user",
            "content": "first",
        },
    ]

    new_turns = [
        {
            "role": "assistant",
            "content": "second",
        },
        {
            "role": "user",
            "content": "third",
        },
    ]

    result = _add_and_cap_history(
        existing,
        new_turns,
    )

    assert result == [
        {
            "role": "user",
            "content": "first",
        },
        {
            "role": "assistant",
            "content": "second",
        },
        {
            "role": "user",
            "content": "third",
        },
    ]