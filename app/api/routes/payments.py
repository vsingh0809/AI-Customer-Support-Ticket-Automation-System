"""Payment API routes."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.api.schemas.payment import PaymentResponse
from app.core.exceptions import ApplicationServiceError
from app.services.order_service import OrderService
from app.services.payment_service import PaymentService

router = APIRouter(prefix="/payments", tags=["payments"])


def get_order_service(
    db: Annotated[Session, Depends(get_db)],
) -> OrderService:
    """Provide an order service for the current database session."""
    return OrderService(db)


def get_payment_service(
    db: Annotated[Session, Depends(get_db)],
) -> PaymentService:
    """Provide a payment service for the current database session."""
    return PaymentService(db)


@router.get(
    "/{order_id}",
    response_model=PaymentResponse,
)
def get_payment(
    order_id: Annotated[str, Path(min_length=1, max_length=50)],
    customer_id:UUID,
    order_service: Annotated[
        OrderService,
        Depends(get_order_service),
    ],
    payment_service: Annotated[
        PaymentService,
        Depends(get_payment_service),
    ],
) -> PaymentResponse:
    """Get the latest payment for a customer's order."""
    normalized_order_id = order_id.strip()

    if not normalized_order_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Order ID is required.",
        )

    try:
        order = order_service.get_by_external_id(
            normalized_order_id,
            customer_id,
        )
    except ApplicationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve order.",
        ) from exc

    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found.",
        )

    try:
        payment = payment_service.get_latest_for_order(order.id)
    except ApplicationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve payment.",
        ) from exc

    if payment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found.",
        )

    return PaymentResponse(
        payment_id=payment.id,
        order_id=normalized_order_id,
        transaction_id=payment.transaction_id,
        status=payment.status,
        amount=str(payment.amount),
        created_at=(
            payment.created_at.isoformat()
            if payment.created_at
            else None
        ),
    )