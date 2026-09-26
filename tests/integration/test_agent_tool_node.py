from unittest.mock import Mock
from uuid import uuid4

from app.ai.agent.contracts import Intent
from app.ai.agent.nodes import (
    build_tool_node,
    build_tool_selection_node,
)
from app.ai.tools.base import ToolResult
from app.ai.tools.registry import ToolRegistry


def test_order_intent_selects_order_tool() -> None:
    node = build_tool_selection_node()

    state = {
        "intent": Intent.ORDER_STATUS,
        "errors": [],
    }

    result = node(state)

    assert result["tool_name"] == "check_order_status"


def test_payment_intent_selects_payment_tool() -> None:
    node = build_tool_selection_node()

    state = {
        "intent": Intent.PAYMENT_STATUS,
        "errors": [],
    }

    result = node(state)

    assert result["tool_name"] == "check_payment_status"


def test_unsupported_intent_does_not_select_tool() -> None:
    node = build_tool_selection_node()

    state = {
        "intent": Intent.KNOWLEDGE_QUERY,
        "errors": [],
    }

    result = node(state)

    assert result["tool_name"] is None
    assert result["errors"]


def test_tool_node_executes_registered_tool() -> None:
    tool = Mock()
    tool.name = "check_order_status"
    tool.execute.return_value = ToolResult.ok(
        {
            "order_id": "45821",
            "status": "shipped",
        }
    )

    registry = ToolRegistry()
    registry.register(tool)

    node = build_tool_node(registry)

    customer_id = uuid4()
    conversation_id = uuid4()

    state = {
        "tool_name": "check_order_status",
        "tool_arguments": {
            "order_id": "45821",
        },
        "customer_id": customer_id,
        "conversation_id": conversation_id,
        "errors": [],
    }

    result = node(state)

    assert result["tool_result"]["success"] is True
    assert result["tool_result"]["data"]["order_id"] == "45821"

    tool.execute.assert_called_once_with(
        order_id="45821",
        customer_id=customer_id,
    )


def test_tool_node_overrides_model_customer_id() -> None:
    tool = Mock()
    tool.name = "check_order_status"
    tool.execute.return_value = ToolResult.ok()

    registry = ToolRegistry()
    registry.register(tool)

    node = build_tool_node(registry)

    authenticated_customer_id = uuid4()
    fake_customer_id = uuid4()

    state = {
        "tool_name": "check_order_status",
        "tool_arguments": {
            "order_id": "45821",
            "customer_id": fake_customer_id,
        },
        "customer_id": authenticated_customer_id,
        "errors": [],
    }

    node(state)

    tool.execute.assert_called_once_with(
        order_id="45821",
        customer_id=authenticated_customer_id,
    )


def test_tool_node_handles_unknown_tool() -> None:
    registry = ToolRegistry()

    node = build_tool_node(registry)

    state = {
        "tool_name": "unknown_tool",
        "tool_arguments": {},
        "errors": [],
    }

    result = node(state)

    assert result["tool_result"] is None
    assert result["errors"]