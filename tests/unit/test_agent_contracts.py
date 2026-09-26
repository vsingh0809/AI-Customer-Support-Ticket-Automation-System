from uuid import uuid4

from app.ai.agent.contracts import AgentRoute, Intent
from app.ai.agent.state import AgentState


def test_intent_values_are_stable() -> None:
    assert Intent.KNOWLEDGE_QUERY.value == "knowledge_query"
    assert Intent.ORDER_STATUS.value == "order_status"
    assert Intent.PAYMENT_STATUS.value == "payment_status"
    assert Intent.REFUND.value == "refund"
    assert Intent.CANCELLATION.value == "cancellation"
    assert Intent.SUPPORT_TICKET.value == "support_ticket"
    assert Intent.HUMAN_ESCALATION.value == "human_escalation"
    assert Intent.UNKNOWN.value == "unknown"


def test_agent_route_values_are_stable() -> None:
    assert AgentRoute.RAG.value == "rag"
    assert AgentRoute.TOOL.value == "tool"
    assert AgentRoute.ASK_CLARIFICATION.value == "ask_clarification"
    assert AgentRoute.TICKET.value == "ticket"
    assert AgentRoute.ESCALATE.value == "escalate"
    assert AgentRoute.RESPOND.value == "respond"


def test_agent_state_can_hold_expected_fields() -> None:
    state: AgentState = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "Where is my order 45821?",
        "intent": Intent.ORDER_STATUS,
        "route": AgentRoute.TOOL,
        "entities": {
            "order_id": "45821",
        },
        "retrieved_context": [],
        "sources": [],
        "tool_name": "check_order_status",
        "tool_arguments": {
            "order_id": "45821",
        },
        "tool_result": None,
        "response": None,
        "ticket_id": None,
        "escalated": False,
        "errors": [],
    }

    assert state["intent"] == Intent.ORDER_STATUS
    assert state["route"] == AgentRoute.TOOL
    assert state["entities"]["order_id"] == "45821"