"""Human-escalation tools exposed to the AI agent."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import ValidationError

from app.ai.tools.base import ToolResult
from app.ai.tools.schemas import EscalationInput
from app.core.exceptions import ApplicationServiceError
from app.services.ticket_service import TicketService

_ALLOWED_PRIORITIES = {
    "high",
    "urgent",
}


class EscalationTools:
    """Business tools for escalating customer issues to humans."""
    name = "escalate_to_human"

    description = (
        "Escalate a customer issue to human support when AI handling "
        "is insufficient or the customer explicitly requests a human."
    )

    def __init__(self, ticket_service: TicketService) -> None:
        self._ticket_service = ticket_service

    def escalate_to_human(
        self,
        *,
        customer_id: UUID | None,
        reason: str,
        conversation_id: UUID | None = None,
        priority: str = "urgent",
    ) -> ToolResult:
        """Create a support ticket for human intervention."""
        if customer_id is None:
            return ToolResult.failure(
                error_code="INVALID_CUSTOMER_ID",
                error_message="Customer ID is required for escalation.",
            )

        if not reason or not reason.strip():
            return ToolResult.failure(
                error_code="INVALID_ESCALATION_REASON",
                error_message="Escalation reason is required.",
            )

        normalized_reason = reason.strip()
        normalized_priority = priority.strip().lower()

        if normalized_priority not in _ALLOWED_PRIORITIES:
            return ToolResult.failure(
                error_code="INVALID_ESCALATION_PRIORITY",
                error_message=(
                    "Escalation priority must be either "
                    "high or urgent."
                ),
            )

        try:
            ticket = self._ticket_service.create(
                customer_id=customer_id,
                category="human_escalation",
                description=normalized_reason,
                priority=normalized_priority,
                conversation_id=conversation_id,
            )
        except ApplicationServiceError:
            return ToolResult.failure(
                error_code="ESCALATION_FAILED",
                error_message=(
                    "Unable to escalate the issue to human support "
                    "right now."
                ),
            )

        return ToolResult.ok(
            {
                "escalated": True,
                "ticket_id": str(ticket.id),
                "category": ticket.category,
                "priority": ticket.priority,
                "status": ticket.status,
                "reason": ticket.description,
                "conversation_id": (
                    str(ticket.conversation_id)
                    if ticket.conversation_id
                    else None
                ),
            }
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Validate and execute human escalation."""
        try:
            params = EscalationInput.model_validate(kwargs)
        except ValidationError:
            return ToolResult.failure(
                error_code="INVALID_TOOL_PARAMETERS",
                error_message="Invalid parameters for escalate_to_human.",
            )

        return self.escalate_to_human(
            customer_id=params.customer_id,
            reason=params.reason,
            conversation_id=params.conversation_id,
            priority=params.priority,
        )