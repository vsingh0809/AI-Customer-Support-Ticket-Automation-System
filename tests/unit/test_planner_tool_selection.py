from app.ai.agent.contracts import Intent
from app.ai.agent.nodes import build_tool_selection_node
from app.ai.agent.state import ActionType


def test_tool_selection_uses_planned_tool() -> None:
    node = build_tool_selection_node()

    state = {
        "intent": Intent.ORDER_STATUS,
        "planned_actions": [
            {
                "intent": Intent.ORDER_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": "check_order_status",
            }
        ],
        "errors": [],
    }

    result = node(state)

    assert result["tool_name"] == "check_order_status"
    assert result.get("errors", []) == []


def test_tool_selection_ignores_conflicting_legacy_intent() -> None:
    node = build_tool_selection_node()

    state = {
        "intent": Intent.ORDER_STATUS,
        "planned_actions": [
            {
                "intent": Intent.PAYMENT_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": "check_payment_status",
            }
        ],
        "errors": [],
    }

    result = node(state)

    assert result["tool_name"] == "check_payment_status"
    assert result.get("errors", []) == []


def test_tool_selection_rejects_multiple_planned_actions() -> None:
    node = build_tool_selection_node()

    state = {
        "intent": Intent.ORDER_STATUS,
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
        "errors": [],
    }

    result = node(state)

    assert result["tool_name"] is None
    assert result["errors"]


def test_tool_selection_rejects_non_tool_action() -> None:
    node = build_tool_selection_node()

    state = {
        "intent": Intent.REFUND,
        "planned_actions": [
            {
                "intent": Intent.REFUND,
                "action_type": ActionType.RAG,
                "tool_name": None,
            }
        ],
        "errors": [],
    }

    result = node(state)

    assert result["tool_name"] is None
    assert result["errors"]


def test_tool_selection_rejects_missing_tool_name() -> None:
    node = build_tool_selection_node()

    state = {
        "intent": Intent.ORDER_STATUS,
        "planned_actions": [
            {
                "intent": Intent.ORDER_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": None,
            }
        ],
        "errors": [],
    }

    result = node(state)

    assert result["tool_name"] is None
    assert result["errors"]