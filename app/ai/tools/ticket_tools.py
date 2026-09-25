"""Support-ticket tools exposed to the AI agent."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import ValidationError

from app.ai.tools.base import ToolResult
from app.ai.tools.schemas import CreateTicketInput
from app.core.exceptions import ApplicationServiceError
from app.services.ticket_service import TicketService

_ALLOWED_PRIORITIES = {
    "low",
    "normal",
    "high",
    "urgent",
}


class TicketTools:
    """Business tools for support-ticket operations."""

    tool_name = "create_support_ticket"

    description = (
        "Create a support ticket for a customer issue that requires "
        "support-team handling."
    )

    def __init__(self, ticket_service: TicketService) -> None:
        self._ticket_service = ticket_service

    def create_support_ticket(
        self,
        *,
        customer_id: UUID | None,
        category: str,
        description: str,
        priority: str = "normal",
        conversation_id: UUID | None = None,
    ) -> ToolResult:
        """Create a support ticket for a customer."""
        if customer_id is None:
            return ToolResult.failure(
                error_code="INVALID_CUSTOMER_ID",
                error_message="Customer ID is required.",
            )

        if not category or not category.strip():
            return ToolResult.failure(
                error_code="INVALID_TICKET_CATEGORY",
                error_message="Ticket category is required.",
            )

        if not description or not description.strip():
            return ToolResult.failure(
                error_code="INVALID_TICKET_DESCRIPTION",
                error_message="Ticket description is required.",
            )

        normalized_category = category.strip()
        normalized_description = description.strip()
        normalized_priority = priority.strip().lower()

        if normalized_priority not in _ALLOWED_PRIORITIES:
            return ToolResult.failure(
                error_code="INVALID_TICKET_PRIORITY",
                error_message=(
                    "Ticket priority must be one of: "
                    "low, normal, high, urgent."
                ),
            )

        try:
            ticket = self._ticket_service.create(
                customer_id=customer_id,
                category=normalized_category,
                description=normalized_description,
                priority=normalized_priority,
                conversation_id=conversation_id,
            )
        except ApplicationServiceError:
            return ToolResult.failure(
                error_code="TICKET_CREATION_FAILED",
                error_message=(
                    "Unable to create a support ticket right now."
                ),
            )

        return ToolResult.ok(
            {
                "ticket_id": str(ticket.id),
                "customer_id": str(ticket.customer_id),
                "category": ticket.category,
                "priority": ticket.priority,
                "status": ticket.status,
                "description": ticket.description,
                "conversation_id": (
                    str(ticket.conversation_id)
                    if ticket.conversation_id
                    else None
                ),
                "created_at": (
                    ticket.created_at.isoformat()
                    if ticket.created_at
                    else None
                ),
            }
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Validate and execute ticket creation."""
        try:
            params = CreateTicketInput.model_validate(kwargs)
        except ValidationError:
            return ToolResult.failure(
                error_code="INVALID_TOOL_PARAMETERS",
                error_message="Invalid parameters for create_support_ticket.",
            )

        return self.create_support_ticket(
            customer_id=params.customer_id,
            category=params.category,
            description=params.description,
            priority=params.priority,
            conversation_id=params.conversation_id,
        )