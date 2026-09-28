from app.ai.agent.contracts import AgentRoute, Intent
from app.ai.agent.graph import _route_after_intent
from app.ai.agent.state import ActionType


def test_planned_rag_action_controls_routing() -> None:
    state = {
        "planned_actions": [
            {
                "intent": Intent.REFUND,
                "action_type": ActionType.RAG,
                "tool_name": None,
            }
        ],
        # Deliberately conflicting legacy route.
        "route": AgentRoute.TOOL,
    }

    assert _route_after_intent(state) == "rag"


def test_planned_single_tool_action_controls_routing() -> None:
    state = {
        "planned_actions": [
            {
                "intent": Intent.ORDER_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": "check_order_status",
            }
        ],
        # Deliberately conflicting legacy route.
        "route": AgentRoute.RAG,
    }

    assert _route_after_intent(state) == "tool"


def test_multiple_planned_tool_actions_route_to_multi_tool() -> None:
    state = {
        "planned_actions": [
            {
                "intent": Intent.ORDER_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": "check_order_status",
            },
            {
                "intent": Intent.PAYMENT_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": "check_payment_status",
            },
        ],
        # Deliberately conflicting legacy route.
        "route": AgentRoute.RAG,
    }

    assert _route_after_intent(state) == "multi_tool"


def test_mixed_planned_actions_route_to_mixed_action() -> None:
    state = {
        "planned_actions": [
            {
                "intent": Intent.REFUND,
                "action_type": ActionType.RAG,
                "tool_name": None,
            },
            {
                "intent": Intent.ORDER_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": "check_order_status",
            },
        ],
        # Deliberately conflicting legacy route.
        "route": AgentRoute.TOOL,
    }

    assert _route_after_intent(state) == "mixed_action"


def test_mixed_planned_actions_preserve_reverse_order() -> None:
    state = {
        "planned_actions": [
            {
                "intent": Intent.ORDER_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": "check_order_status",
            },
            {
                "intent": Intent.REFUND,
                "action_type": ActionType.RAG,
                "tool_name": None,
            },
        ],
        "route": AgentRoute.RAG,
    }

    assert _route_after_intent(state) == "mixed_action"
    assert state["planned_actions"][0]["action_type"] == (
        ActionType.TOOL
    )
    assert state["planned_actions"][1]["action_type"] == (
        ActionType.RAG
    )


def test_no_planned_actions_preserves_clarification_route() -> None:
    state = {
        "planned_actions": [],
        "route": AgentRoute.ASK_CLARIFICATION,
    }

    assert _route_after_intent(state) == "ask_clarification"


def test_no_planned_actions_end_when_no_usable_route_exists() -> None:
    state = {
        "planned_actions": [],
        "route": None,
    }

    assert _route_after_intent(state) == "end"