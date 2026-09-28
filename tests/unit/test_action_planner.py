from app.ai.agent.contracts import Intent
from app.ai.agent.nodes import build_action_planning_node
from app.ai.agent.state import ActionType


def test_action_planner_maps_tool_intent_to_planned_tool() -> None:
    node = build_action_planning_node()

    result = node(
        {
            "intents": [Intent.ORDER_STATUS],
            "errors": [],
        }
    )

    assert result["planned_actions"] == [
        {
            "intent": Intent.ORDER_STATUS,
            "action_type": ActionType.TOOL,
            "tool_name": "check_order_status",
        }
    ]


def test_action_planner_maps_rag_intent_to_planned_rag() -> None:
    node = build_action_planning_node()

    result = node(
        {
            "intents": [Intent.REFUND],
            "errors": [],
        }
    )

    assert result["planned_actions"] == [
        {
            "intent": Intent.REFUND,
            "action_type": ActionType.RAG,
            "tool_name": None,
        }
    ]


def test_action_planner_maps_all_tool_intents() -> None:
    node = build_action_planning_node()

    result = node(
        {
            "intents": [
                Intent.ORDER_STATUS,
                Intent.PAYMENT_STATUS,
                Intent.SUPPORT_TICKET,
                Intent.HUMAN_ESCALATION,
            ],
            "errors": [],
        }
    )

    assert result["planned_actions"] == [
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
        {
            "intent": Intent.SUPPORT_TICKET,
            "action_type": ActionType.TOOL,
            "tool_name": "create_support_ticket",
        },
        {
            "intent": Intent.HUMAN_ESCALATION,
            "action_type": ActionType.TOOL,
            "tool_name": "escalate_to_human",
        },
    ]


def test_action_planner_preserves_intent_order() -> None:
    node = build_action_planning_node()

    result = node(
        {
            "intents": [
                Intent.REFUND,
                Intent.ORDER_STATUS,
                Intent.PAYMENT_STATUS,
            ],
            "errors": [],
        }
    )

    assert [
        action["intent"]
        for action in result["planned_actions"]
    ] == [
        Intent.REFUND,
        Intent.ORDER_STATUS,
        Intent.PAYMENT_STATUS,
    ]

    assert [
        action["action_type"]
        for action in result["planned_actions"]
    ] == [
        ActionType.RAG,
        ActionType.TOOL,
        ActionType.TOOL,
    ]

    assert [
        action["tool_name"]
        for action in result["planned_actions"]
    ] == [
        None,
        "check_order_status",
        "check_payment_status",
    ]


def test_action_planner_ignores_unknown_intent_as_executable_action() -> None:
    node = build_action_planning_node()

    result = node(
        {
            "intents": [Intent.UNKNOWN],
            "errors": [],
        }
    )

    assert result["planned_actions"] == []


def test_action_planner_supports_legacy_single_intent_state() -> None:
    node = build_action_planning_node()

    result = node(
        {
            "intent": Intent.PAYMENT_STATUS,
            "errors": [],
        }
    )

    assert result["planned_actions"] == [
        {
            "intent": Intent.PAYMENT_STATUS,
            "action_type": ActionType.TOOL,
            "tool_name": "check_payment_status",
        }
    ]


def test_action_planner_rejects_missing_classification() -> None:
    node = build_action_planning_node()

    result = node(
        {
            "errors": [],
        }
    )

    assert result["planned_actions"] == []
    assert result["errors"] == [
        "Cannot plan actions without classified intents"
    ]

