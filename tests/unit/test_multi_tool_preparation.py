from types import SimpleNamespace
from unittest.mock import Mock

from app.ai.agent.contracts import Intent
from app.ai.agent.nodes import build_multi_tool_preparation_node
from app.ai.agent.state import ActionType


def test_multi_tool_preparation_uses_planned_tool_names() -> None:
    extractor = Mock()

    extractor.prepare.side_effect = [
        SimpleNamespace(
            arguments={"order_id": "45821"},
            missing_fields=(),
            clarification_question=None,
        ),
        SimpleNamespace(
            arguments={"order_id": "45821"},
            missing_fields=(),
            clarification_question=None,
        ),
    ]

    node = build_multi_tool_preparation_node(extractor)

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
        "user_message": "Check order 45821 and its payment.",
        "conversation_history": [],
        "errors": [],
    }

    result = node(state)

    assert result["planned_tool_actions"][0]["tool_name"] == (
        "check_order_status"
    )
    assert result["planned_tool_actions"][1]["tool_name"] == (
        "check_payment_status"
    )


def test_multi_tool_preparation_preserves_planned_action_order() -> None:
    extractor = Mock()

    extractor.prepare.side_effect = [
        SimpleNamespace(
            arguments={"order_id": "45821"},
            missing_fields=(),
            clarification_question=None,
        ),
        SimpleNamespace(
            arguments={"order_id": "45821"},
            missing_fields=(),
            clarification_question=None,
        ),
    ]

    node = build_multi_tool_preparation_node(extractor)

    state = {
        "planned_actions": [
            {
                "intent": Intent.PAYMENT_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": "check_payment_status",
            },
            {
                "intent": Intent.ORDER_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": "check_order_status",
            },
        ],
        "user_message": "Check payment and order 45821.",
        "conversation_history": [],
        "errors": [],
    }

    result = node(state)

    assert [
        action["tool_name"]
        for action in result["planned_tool_actions"]
    ] == [
        "check_payment_status",
        "check_order_status",
    ]


def test_multi_tool_preparation_rejects_non_tool_action() -> None:
    extractor = Mock()
    node = build_multi_tool_preparation_node(extractor)

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
        "user_message": "Tell me the refund policy and check order 45821.",
        "conversation_history": [],
        "errors": [],
    }

    result = node(state)

    assert result["planned_tool_actions"] == []
    assert result["errors"]


def test_multi_tool_preparation_requires_at_least_two_actions() -> None:
    extractor = Mock()
    node = build_multi_tool_preparation_node(extractor)

    state = {
        "planned_actions": [
            {
                "intent": Intent.ORDER_STATUS,
                "action_type": ActionType.TOOL,
                "tool_name": "check_order_status",
            }
        ],
        "user_message": "Check order 45821.",
        "conversation_history": [],
        "errors": [],
    }

    result = node(state)

    assert result["planned_tool_actions"] == []
    assert result["errors"]


def test_multi_tool_preparation_preserves_missing_fields() -> None:
    extractor = Mock()

    extractor.prepare.side_effect = [
        SimpleNamespace(
            arguments={},
            missing_fields=("order_id",),
            clarification_question=(
                "Please provide your order ID so I can check it."
            ),
        ),
        SimpleNamespace(
            arguments={"order_id": "45821"},
            missing_fields=(),
            clarification_question=None,
        ),
    ]

    node = build_multi_tool_preparation_node(extractor)

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
        "user_message": "Check my order and payment.",
        "conversation_history": [],
        "errors": [],
    }

    result = node(state)

    assert result["missing_fields"] == [
        "order_status.order_id",
    ]

    assert result["clarification_question"] == (
        "I need your order ID to complete the order-status and "
        "payment-status checks. For example: 45821."
    )