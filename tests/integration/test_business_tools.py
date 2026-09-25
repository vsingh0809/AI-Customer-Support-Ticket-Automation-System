from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from app.ai.tools.base import ToolResult
from app.ai.tools.escalation_tools import EscalationTools
from app.ai.tools.order_tools import OrderTools
from app.ai.tools.payment_tools import PaymentTools
from app.ai.tools.ticket_tools import TicketTools
from app.core.exceptions import ApplicationServiceError


def _build_order():
    return SimpleNamespace(
        id=uuid4(),
        external_order_id="45821",
        status="shipped",
        total_amount="1499.00",
        expected_delivery=None,
    )


def _build_payment(order_id):
    return SimpleNamespace(
        id=uuid4(),
        order_id=order_id,
        transaction_id="txn-45821",
        status="successful",
        amount="1499.00",
        created_at=None,
    )


def _build_ticket(
    *,
    category: str,
    priority: str,
    description: str,
):
    return SimpleNamespace(
        id=uuid4(),
        customer_id=uuid4(),
        category=category,
        priority=priority,
        status="open",
        description=description,
        conversation_id=uuid4(),
        created_at=None,
    )


def test_order_then_payment_workflow() -> None:
    order_service = Mock()
    payment_service = Mock()

    order = _build_order()
    payment = _build_payment(order.id)

    order_service.get_by_external_id.return_value = order
    payment_service.get_latest_for_order.return_value = payment

    order_tool = OrderTools(order_service)

    payment_tool = PaymentTools(
        order_service=order_service,
        payment_service=payment_service,
    )

    order_result = order_tool.check_order_status("45821")

    assert isinstance(order_result, ToolResult)
    assert order_result.success is True
    assert order_result.data["order_id"] == "45821"

    payment_result = payment_tool.check_payment_status("45821")

    assert isinstance(payment_result, ToolResult)
    assert payment_result.success is True
    assert payment_result.data["order_id"] == "45821"
    assert payment_result.data["payment_id"] == str(payment.id)

    assert order_service.get_by_external_id.call_count == 2

    payment_service.get_latest_for_order.assert_called_once_with(
        order.id,
    )


def test_ticket_and_escalation_use_same_ticket_service_contract() -> None:
    ticket_service = Mock()

    support_ticket = _build_ticket(
        category="payment",
        priority="high",
        description="Payment was deducted but order failed.",
    )

    escalation_ticket = _build_ticket(
        category="human_escalation",
        priority="urgent",
        description="Customer requested human support.",
    )

    ticket_service.create.side_effect = [
        support_ticket,
        escalation_ticket,
    ]

    ticket_tool = TicketTools(ticket_service)
    escalation_tool = EscalationTools(ticket_service)

    support_result = ticket_tool.create_support_ticket(
        customer_id=support_ticket.customer_id,
        category="payment",
        description=support_ticket.description,
        priority="high",
        conversation_id=support_ticket.conversation_id,
    )

    escalation_result = escalation_tool.escalate_to_human(
        customer_id=escalation_ticket.customer_id,
        reason=escalation_ticket.description,
        conversation_id=escalation_ticket.conversation_id,
    )

    assert isinstance(support_result, ToolResult)
    assert isinstance(escalation_result, ToolResult)

    assert support_result.success is True
    assert support_result.data["category"] == "payment"

    assert escalation_result.success is True
    assert escalation_result.data["escalated"] is True
    assert escalation_result.data["category"] == "human_escalation"

    assert ticket_service.create.call_count == 2


def test_payment_workflow_preserves_customer_boundary() -> None:
    order_service = Mock()
    payment_service = Mock()

    order = _build_order()
    payment = _build_payment(order.id)
    customer_id = uuid4()

    order_service.get_by_external_id.return_value = order
    payment_service.get_latest_for_order.return_value = payment

    tool = PaymentTools(
        order_service=order_service,
        payment_service=payment_service,
    )

    result = tool.check_payment_status(
        order_id="45821",
        customer_id=customer_id,
    )

    assert result.success is True

    order_service.get_by_external_id.assert_called_once_with(
        "45821",
        customer_id,
    )

    payment_service.get_latest_for_order.assert_called_once_with(
        order.id,
    )


def test_tool_failure_does_not_leak_internal_exception() -> None:
    order_service = Mock()
    payment_service = Mock()

    order_service.get_by_external_id.side_effect = ApplicationServiceError(
        "database connection refused",
    )

    payment_tool = PaymentTools(
        order_service=order_service,
        payment_service=payment_service,
    )

    result = payment_tool.check_payment_status("45821")

    assert result.success is False
    assert result.error_code == "PAYMENT_LOOKUP_FAILED"

    assert result.error_message is not None
    assert "database connection refused" not in result.error_message