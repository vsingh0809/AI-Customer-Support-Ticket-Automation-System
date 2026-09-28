"""Support-ticket API routes."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.api.schemas.ticket import CreateTicketRequest, TicketResponse
from app.core.exceptions import ApplicationServiceError
from app.services.ticket_service import TicketService

router = APIRouter(prefix="/tickets", tags=["tickets"])


def get_ticket_service(
    db: Annotated[Session, Depends(get_db)],
) -> TicketService:
    """Provide a ticket service for the current database session."""
    return TicketService(db)


@router.post(
    "",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_ticket(
    request: CreateTicketRequest,
    service: Annotated[
        TicketService,
        Depends(get_ticket_service),
    ],
) -> TicketResponse:
    """Create a support ticket."""
    try:
        ticket = service.create(
            customer_id=request.customer_id,
            category=request.category.strip(),
            description=request.description.strip(),
            priority=request.priority,
            conversation_id=request.conversation_id,
        )
    except ApplicationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to create support ticket.",
        ) from exc

    return TicketResponse.model_validate(ticket)

@router.get(
    "/{ticket_id}",
    response_model=TicketResponse,
)
def get_ticket(
    ticket_id: UUID,
    customer_id: UUID,
    service: Annotated[
        TicketService,
        Depends(get_ticket_service),
    ],
) -> TicketResponse:
    """Get a customer's support ticket."""
    try:
        ticket = service.get_by_id(
            ticket_id=ticket_id,
            customer_id=customer_id,
        )
    except ApplicationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve support ticket.",
        ) from exc

    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Support ticket not found.",
        )

    return TicketResponse.model_validate(ticket)