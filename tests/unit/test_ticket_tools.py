from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from app.ai.tools.ticket_tools import TicketTools
from app.core.exceptions import ApplicationServiceError


def _build_ticket():
    return SimpleNamespace(
        id=uuid4(),
        customer_id=uuid4(),
        category="payment",
        priority="high",
        status="open",
        description="Payment was deducted but order failed.",
        conversation_id=uuid4(),
        created_at=None,
    )


def test_create_support_ticket_success() -> None:
    service = Mock()

    ticket = _build_ticket()
    service.create.return_value = ticket

    tool = TicketTools(service)

    result = tool.create_support_ticket(
        customer_id=ticket.customer_id,
        category=" payment ",
        description=" Payment was deducted but order failed. ",
        priority="HIGH",
        conversation_id=ticket.conversation_id,
    )

    assert result.success is True
    assert result.data["ticket_id"] == str(ticket.id)
    assert result.data["customer_id"] == str(ticket.customer_id)
    assert result.data["category"] == "payment"
    assert result.data["priority"] == "high"
    assert result.data["status"] == "open"
    assert result.data["conversation_id"] == str(ticket.conversation_id)

    service.create.assert_called_once_with(
        customer_id=ticket.customer_id,
        category="payment",
        description="Payment was deducted but order failed.",
        priority="high",
        conversation_id=ticket.conversation_id,
    )


def test_create_support_ticket_rejects_missing_customer() -> None:
    service = Mock()
    tool = TicketTools(service)

    result = tool.create_support_ticket(
        customer_id=None,
        category="payment",
        description="Payment issue",
    )

    assert result.success is False
    assert result.error_code == "INVALID_CUSTOMER_ID"
    service.create.assert_not_called()


def test_create_support_ticket_rejects_empty_category() -> None:
    service = Mock()
    tool = TicketTools(service)

    result = tool.create_support_ticket(
        customer_id=uuid4(),
        category="   ",
        description="Payment issue",
    )

    assert result.success is False
    assert result.error_code == "INVALID_TICKET_CATEGORY"
    service.create.assert_not_called()


def test_create_support_ticket_rejects_empty_description() -> None:
    service = Mock()
    tool = TicketTools(service)

    result = tool.create_support_ticket(
        customer_id=uuid4(),
        category="payment",
        description="   ",
    )

    assert result.success is False
    assert result.error_code == "INVALID_TICKET_DESCRIPTION"
    service.create.assert_not_called()


def test_create_support_ticket_rejects_invalid_priority() -> None:
    service = Mock()
    tool = TicketTools(service)

    result = tool.create_support_ticket(
        customer_id=uuid4(),
        category="payment",
        description="Payment issue",
        priority="critical",
    )

    assert result.success is False
    assert result.error_code == "INVALID_TICKET_PRIORITY"
    service.create.assert_not_called()


def test_create_support_ticket_handles_service_failure() -> None:
    service = Mock()

    service.create.side_effect = ApplicationServiceError(
        "Failed to create support ticket"
    )

    tool = TicketTools(service)

    result = tool.create_support_ticket(
        customer_id=uuid4(),
        category="payment",
        description="Payment issue",
    )

    assert result.success is False
    assert result.error_code == "TICKET_CREATION_FAILED"