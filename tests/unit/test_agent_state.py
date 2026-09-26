from uuid import uuid4

from app.ai.agent.state import AgentState


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