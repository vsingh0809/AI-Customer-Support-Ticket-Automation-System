from unittest.mock import Mock

from app.ai.agent.nodes import build_finalization_node


def test_finalization_runs_guardrails_and_tracing() -> None:
    tracer = Mock()
    guardrails = Mock()

    node = build_finalization_node(
        tracer=tracer,
        guardrails=guardrails,
    )

    state = {
        "user_message": "Where is my order?",
        "response": "Your order is shipped.",
        "errors": [],
    }

    result = node(state)

    assert result == {}

    guardrails.check.assert_called_once_with(state)
    tracer.on_turn_end.assert_called_once_with(state)