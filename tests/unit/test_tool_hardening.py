from unittest.mock import Mock
from uuid import uuid4

from app.ai.tools.escalation_tools import EscalationTools
from app.ai.tools.order_tools import OrderTools
from app.ai.tools.payment_tools import PaymentTools
from app.ai.tools.ticket_tools import TicketTools


def test_order_tool_rejects_invalid_parameters() -> None:
    service = Mock()

    tool = OrderTools(service)

    result = tool.execute(
        order_id="45821",
        unexpected_parameter="blocked",
    )

    assert result.success is False
    assert result.error_code == "INVALID_TOOL_PARAMETERS"
    service.get_by_external_id.assert_not_called()


def test_order_tool_requires_customer_id() -> None:
    service = Mock()

    tool = OrderTools(service)

    result = tool.execute(
        order_id="45821",
    )

    assert result.success is False
    assert result.error_code == "INVALID_TOOL_PARAMETERS"
    service.get_by_external_id.assert_not_called()


def test_order_tool_rejects_invalid_customer_id() -> None:
    service = Mock()

    tool = OrderTools(service)

    result = tool.execute(
        order_id="45821",
        customer_id="not-a-uuid",
    )

    assert result.success is False
    assert result.error_code == "INVALID_TOOL_PARAMETERS"
    service.get_by_external_id.assert_not_called()


def test_payment_tool_rejects_invalid_parameters() -> None:
    order_service = Mock()
    payment_service = Mock()

    tool = PaymentTools(
        order_service=order_service,
        payment_service=payment_service,
    )

    result = tool.execute(
        order_id="",
        customer_id=str(uuid4()),
    )

    assert result.success is False
    assert result.error_code == "INVALID_TOOL_PARAMETERS"

    order_service.get_by_external_id.assert_not_called()
    payment_service.get_latest_for_order.assert_not_called()


def test_ticket_tool_rejects_unknown_parameter() -> None:
    service = Mock()

    tool = TicketTools(service)

    result = tool.execute(
        customer_id=str(uuid4()),
        category="payment",
        description="Payment issue",
        unknown_field="blocked",
    )

    assert result.success is False
    assert result.error_code == "INVALID_TOOL_PARAMETERS"

    service.create.assert_not_called()


def test_ticket_tool_rejects_oversized_description() -> None:
    service = Mock()

    tool = TicketTools(service)

    result = tool.execute(
        customer_id=str(uuid4()),
        category="payment",
        description="x" * 2001,
    )

    assert result.success is False
    assert result.error_code == "INVALID_TOOL_PARAMETERS"

    service.create.assert_not_called()


def test_escalation_tool_rejects_invalid_priority() -> None:
    service = Mock()

    tool = EscalationTools(service)

    result = tool.execute(
        customer_id=str(uuid4()),
        reason="Customer explicitly requested human support.",
        priority="normal",
    )

    assert result.success is False
    assert result.error_code == "INVALID_TOOL_PARAMETERS"

    service.create.assert_not_called()


def test_escalation_tool_accepts_string_uuid() -> None:
    service = Mock()

    ticket = Mock()
    ticket.id = uuid4()
    ticket.customer_id = uuid4()
    ticket.category = "human_escalation"
    ticket.priority = "urgent"
    ticket.status = "open"
    ticket.description = "Customer requested human support."
    ticket.conversation_id = None

    service.create.return_value = ticket

    tool = EscalationTools(service)

    customer_id = uuid4()

    result = tool.execute(
        customer_id=str(customer_id),
        reason="Customer requested human support.",
    )

    assert result.success is True
    assert result.data["escalated"] is True

def test_tool_metadata_is_defined() -> None:
    assert OrderTools.tool_name == "check_order_status"
    assert OrderTools.description

    assert PaymentTools.tool_name == "check_payment_status"
    assert PaymentTools.description

    assert TicketTools.tool_name == "create_support_ticket"
    assert TicketTools.description

    assert EscalationTools.tool_name == "escalate_to_human"
    assert EscalationTools.description    