from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from app.ai.agent.argument_extractor import ArgumentExtractor
from app.ai.agent.contracts import Intent
from app.ai.agent.graph import build_agent_graph
from app.ai.agent.intent_classifier import IntentClassifier
from app.ai.agent.router import AgentRouter
from app.ai.tools.escalation_tools import EscalationTools
from app.ai.tools.order_tools import OrderTools
from app.ai.tools.payment_tools import PaymentTools
from app.ai.tools.registry import ToolRegistry
from app.ai.tools.ticket_tools import TicketTools


class FakeStructuredProvider:
    def __init__(
        self,
        *,
        intent: str,
        arguments: dict[str, object],
    ) -> None:
        self.intent = intent
        self.arguments = arguments

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        if "intent classifier" in system_prompt.lower():
            return {"intent": self.intent}

        return self.arguments


def build_real_tool_registry() -> tuple[
    ToolRegistry,
    Mock,
    Mock,
    Mock,
]:
    order_service = Mock()
    payment_service = Mock()
    ticket_service = Mock()

    registry = ToolRegistry()

    registry.register(
        OrderTools(order_service)
    )

    registry.register(
        PaymentTools(
            order_service=order_service,
            payment_service=payment_service,
        )
    )

    registry.register(
        TicketTools(ticket_service)
    )

    registry.register(
        EscalationTools(ticket_service)
    )

    return (
        registry,
        order_service,
        payment_service,
        ticket_service,
    )

def test_graph_executes_real_order_tool() -> None:
    order = SimpleNamespace(
        id=uuid4(),
        external_order_id="45821",
        status="shipped",
        total_amount=Decimal("1499.00"),
        expected_delivery=None,
    )

    registry, order_service, _, _ = build_real_tool_registry()

    order_service.get_by_external_id.return_value = order

    provider = FakeStructuredProvider(
        intent="order_status",
        arguments={
            "order_id": "45821",
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        },
    )

    classifier = IntentClassifier(provider)
    extractor = ArgumentExtractor(provider)

    retriever = Mock()
    generator = Mock()

    graph = build_agent_graph(
        classifier=classifier,
        router=AgentRouter(),
        retriever=retriever,
        generator=generator,
        tool_registry=registry,
        argument_extractor=extractor,
    )

    customer_id = uuid4()
    conversation_id = uuid4()

    result = graph.invoke(
        {
            "conversation_id": conversation_id,
            "customer_id": customer_id,
            "user_message": "Where is my order 45821?",
            "errors": [],
        }
    )

    assert result["intent"] == Intent.ORDER_STATUS
    assert result["tool_name"] == "check_order_status"

    assert result["tool_result"]["success"] is True

    assert result["tool_result"]["data"]["order_id"] == "45821"
    assert result["tool_result"]["data"]["status"] == "shipped"

    order_service.get_by_external_id.assert_called_once_with(
        "45821",
        customer_id,
    )

def test_graph_executes_real_payment_tool() -> None:
    order_id = uuid4()

    order = SimpleNamespace(
        id=order_id,
        external_order_id="45821",
        status="failed",
        total_amount=Decimal("1499.00"),
        expected_delivery=None,
    )

    payment = SimpleNamespace(
        id=uuid4(),
        order_id=order_id,
        transaction_id="txn-45821",
        status="successful",
        amount=Decimal("1499.00"),
        created_at=None,
    )

    registry, order_service, payment_service, _ = (
        build_real_tool_registry()
    )

    order_service.get_by_external_id.return_value = order
    payment_service.get_latest_for_order.return_value = payment

    provider = FakeStructuredProvider(
        intent="payment_status",
        arguments={
            "order_id": "45821",
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        },
    )

    classifier = IntentClassifier(provider)
    extractor = ArgumentExtractor(provider)

    graph = build_agent_graph(
        classifier=classifier,
        router=AgentRouter(),
        retriever=Mock(),
        generator=Mock(),
        tool_registry=registry,
        argument_extractor=extractor,
    )

    customer_id = uuid4()

    result = graph.invoke(
        {
            "conversation_id": uuid4(),
            "customer_id": customer_id,
            "user_message": (
                "What happened to my payment for order 45821?"
            ),
            "errors": [],
        }
    )

    assert result["intent"] == Intent.PAYMENT_STATUS
    assert result["tool_name"] == "check_payment_status"

    assert result["tool_result"]["success"] is True
    assert (
        result["tool_result"]["data"]["payment_id"]
        == str(payment.id)
    )

    order_service.get_by_external_id.assert_called_once_with(
        "45821",
        customer_id,
    )

    payment_service.get_latest_for_order.assert_called_once_with(
        order.id,
    )

def test_graph_executes_real_ticket_tool() -> None:
    customer_id = uuid4()
    conversation_id = uuid4()

    ticket = SimpleNamespace(
        id=uuid4(),
        customer_id=customer_id,
        category="payment",
        priority="high",
        status="open",
        description="Payment was deducted but order failed.",
        conversation_id=conversation_id,
        created_at=None,
    )

    registry, _, _, ticket_service = build_real_tool_registry()

    ticket_service.create.return_value = ticket

    provider = FakeStructuredProvider(
        intent="support_ticket",
        arguments={
            "order_id": None,
            "category": "payment",
            "description": (
                "Payment was deducted but order failed."
            ),
            "reason": None,
            "priority": "high",
        },
    )

    classifier = IntentClassifier(provider)
    extractor = ArgumentExtractor(provider)

    graph = build_agent_graph(
        classifier=classifier,
        router=AgentRouter(),
        retriever=Mock(),
        generator=Mock(),
        tool_registry=registry,
        argument_extractor=extractor,
    )

    result = graph.invoke(
        {
            "conversation_id": conversation_id,
            "customer_id": customer_id,
            "user_message": (
                "Create a support ticket for my payment issue."
            ),
            "errors": [],
        }
    )

    assert result["intent"] == Intent.SUPPORT_TICKET
    assert result["tool_name"] == "create_support_ticket"
    assert result["tool_result"]["success"] is True

    assert (
        result["tool_result"]["data"]["ticket_id"]
        == str(ticket.id)
    )

    ticket_service.create.assert_called_once_with(
        customer_id=customer_id,
        category="payment",
        description="Payment was deducted but order failed.",
        priority="high",
        conversation_id=conversation_id,
    )    
