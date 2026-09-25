from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from app.ai.tools.escalation_tools import EscalationTools
from app.core.exceptions import ApplicationServiceError


def _build_ticket():
    return SimpleNamespace(
        id=uuid4(),
        customer_id=uuid4(),
        category="human_escalation",
        priority="urgent",
        status="open",
        description="Customer explicitly requested human support.",
        conversation_id=uuid4(),
    )


def test_escalate_to_human_success() -> None:
    service = Mock()

    ticket = _build_ticket()
    service.create.return_value = ticket

    tool = EscalationTools(service)

    result = tool.escalate_to_human(
        customer_id=ticket.customer_id,
        reason=ticket.description,
        conversation_id=ticket.conversation_id,
    )

    assert result.success is True
    assert result.data["escalated"] is True
    assert result.data["ticket_id"] == str(ticket.id)
    assert result.data["category"] == "human_escalation"
    assert result.data["priority"] == "urgent"
    assert result.data["status"] == "open"
    assert result.data["conversation_id"] == str(ticket.conversation_id)

    service.create.assert_called_once_with(
        customer_id=ticket.customer_id,
        category="human_escalation",
        description=ticket.description,
        priority="urgent",
        conversation_id=ticket.conversation_id,
    )


def test_escalate_to_human_rejects_missing_customer() -> None:
    service = Mock()
    tool = EscalationTools(service)

    result = tool.escalate_to_human(
        customer_id=None,
        reason="Customer requested a human.",
    )

    assert result.success is False
    assert result.error_code == "INVALID_CUSTOMER_ID"
    service.create.assert_not_called()


def test_escalate_to_human_rejects_empty_reason() -> None:
    service = Mock()
    tool = EscalationTools(service)

    result = tool.escalate_to_human(
        customer_id=uuid4(),
        reason="   ",
    )

    assert result.success is False
    assert result.error_code == "INVALID_ESCALATION_REASON"
    service.create.assert_not_called()


def test_escalate_to_human_rejects_invalid_priority() -> None:
    service = Mock()
    tool = EscalationTools(service)

    result = tool.escalate_to_human(
        customer_id=uuid4(),
        reason="Customer requested human support.",
        priority="normal",
    )

    assert result.success is False
    assert result.error_code == "INVALID_ESCALATION_PRIORITY"
    service.create.assert_not_called()


def test_escalate_to_human_handles_service_failure() -> None:
    service = Mock()

    service.create.side_effect = ApplicationServiceError(
        "Failed to create support ticket"
    )

    tool = EscalationTools(service)

    result = tool.escalate_to_human(
        customer_id=uuid4(),
        reason="Manual investigation is required.",
    )

    assert result.success is False
    assert result.error_code == "ESCALATION_FAILED"


def test_escalate_to_human_allows_high_priority() -> None:
    service = Mock()

    ticket = _build_ticket()
    ticket.priority = "high"

    service.create.return_value = ticket

    customer_id = uuid4()

    tool = EscalationTools(service)

    result = tool.escalate_to_human(
        customer_id=customer_id,
        reason="Complex unresolved complaint.",
        priority="HIGH",
    )

    assert result.success is True

    service.create.assert_called_once_with(
        customer_id=customer_id,
        category="human_escalation",
        description="Complex unresolved complaint.",
        priority="high",
        conversation_id=None,
    )