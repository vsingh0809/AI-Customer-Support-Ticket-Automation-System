"""Human-escalation API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.ai.tools.escalation_tools import EscalationTools
from app.api.dependencies import get_db
from app.api.schemas.escalation import EscalationRequest
from app.api.schemas.ticket import TicketResponse
from app.services.ticket_service import TicketService

router = APIRouter(prefix="/escalate", tags=["escalation"])


def get_escalation_tool(
    db: Annotated[Session, Depends(get_db)],
) -> EscalationTools:
    """Provide the human-escalation tool for the current DB session."""
    return EscalationTools(TicketService(db))


@router.post(
    "",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
)
def escalate(
    request: EscalationRequest,
    tool: Annotated[
        EscalationTools,
        Depends(get_escalation_tool),
    ],
) -> TicketResponse:
    """Escalate a customer issue to human support."""
    result = tool.escalate_to_human(
        customer_id=request.customer_id,
        reason=request.reason.strip(),
        conversation_id=request.conversation_id,
        priority=request.priority,
    )

    if not result.success:
        if result.error_code in {
            "INVALID_CUSTOMER_ID",
            "INVALID_ESCALATION_REASON",
            "INVALID_ESCALATION_PRIORITY",
        }:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message,
            )

        if result.error_code == "ESCALATION_FAILED":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error_message,
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to escalate the issue.",
        )

    data = result.data
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid escalation result.",
        )

    try:
        return TicketResponse(
            id=data["ticket_id"],
            customer_id=request.customer_id,
            category=data["category"],
            description=data["reason"],
            priority=data["priority"],
            status=data["status"],
            conversation_id=(
                request.conversation_id
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid escalation result.",
        ) from exc