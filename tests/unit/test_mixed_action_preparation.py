from types import SimpleNamespace
from unittest.mock import Mock

from app.ai.agent.contracts import Intent
from app.ai.agent.nodes import build_mixed_action_preparation_node
from app.ai.agent.state import ActionType


def test_mixed_preparation_uses_planner_provided_tool_name() -> None:
    extractor = Mock()
    extractor.prepare.return_value = SimpleNamespace(
        arguments={"order_id": "45821"},
        missing_fields=(),
        clarification_question=None,
    )

    node = build_mixed_action_preparation_node(extractor)

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
        "user_message": (
            "Tell me the refund policy and check order 45821."
        ),
        "conversation_history": [],
        "errors": [],
    }

    result = node(state)

    assert result["prepared_actions"][0]["action_type"] == (
        ActionType.RAG
    )
    assert result["prepared_actions"][1]["tool_name"] == (
        "check_order_status"
    )

    extractor.prepare.assert_called_once_with(
        intent=Intent.ORDER_STATUS,
        message="Tell me the refund policy and check order 45821.",
        conversation_history=[],
    )


def test_mixed_preparation_preserves_planned_action_order() -> None:
    extractor = Mock()
    extractor.prepare.return_value = SimpleNamespace(
        arguments={"order_id": "45821"},
        missing_fields=(),
        clarification_question=None,
    )

    node = build_mixed_action_preparation_node(extractor)

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
        "user_message": (
            "Check order 45821 and tell me the refund policy."
        ),
        "conversation_history": [],
        "errors": [],
    }

    result = node(state)

    assert [
        action["action_type"]
        for action in result["prepared_actions"]
    ] == [
        ActionType.TOOL,
        ActionType.RAG,
    ]

    assert result["prepared_actions"][0]["tool_name"] == (
        "check_order_status"
    )
    assert result["prepared_actions"][1]["tool_name"] is None


def test_mixed_preparation_rejects_missing_tool_name() -> None:
    extractor = Mock()
    node = build_mixed_action_preparation_node(extractor)

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
                "tool_name": None,
            },
        ],
        "user_message": (
            "Tell me the refund policy and check my order."
        ),
        "conversation_history": [],
        "errors": [],
    }

    result = node(state)

    assert result["prepared_actions"] == []
    assert result["errors"]


def test_mixed_preparation_rejects_single_action() -> None:
    extractor = Mock()
    node = build_mixed_action_preparation_node(extractor)

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

    assert result["prepared_actions"] == []
    assert result["errors"]


def test_mixed_preparation_preserves_missing_tool_fields() -> None:
    extractor = Mock()
    extractor.prepare.return_value = SimpleNamespace(
        arguments={},
        missing_fields=("order_id",),
        clarification_question=(
            "Please provide your order ID so I can check it."
        ),
    )

    node = build_mixed_action_preparation_node(extractor)

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
        "user_message": (
            "Tell me the refund policy and check my order."
        ),
        "conversation_history": [],
        "errors": [],
    }

    result = node(state)

    assert result["missing_fields"] == [
        "order_status.order_id",
    ]

    assert result["clarification_question"] == (
        "Please provide the missing information."
        if False
        else (
            "I need your order ID. "
            "For example: 45821."
        )
    )


    