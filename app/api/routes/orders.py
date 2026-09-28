"""Order API routes."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.api.schemas.order import OrderResponse
from app.core.exceptions import ApplicationServiceError
from app.services.order_service import OrderService

router = APIRouter(prefix="/orders", tags=["orders"])


def get_order_service(
    db: Annotated[Session, Depends(get_db)],
) -> OrderService:
    """Provide an order service for the current database session."""
    return OrderService(db)


@router.get(
    "/{order_id}",
    response_model=OrderResponse,
)
def get_order(
    order_id: Annotated[str, Path(min_length=1, max_length=50)],
    customer_id: UUID,
    service: Annotated[
        OrderService,
        Depends(get_order_service),
    ],
) -> OrderResponse:
    """Get an order belonging to the requesting customer."""
    normalized_order_id = order_id.strip()

    if not normalized_order_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Order ID is required.",
        )

    try:
        order = service.get_by_external_id(
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

    return OrderResponse(
        id=order.id,
        external_order_id=order.external_order_id,
        customer_id=order.customer_id,
        status=order.status,
        total_amount=str(order.total_amount),
        expected_delivery=(
            order.expected_delivery.isoformat()
            if order.expected_delivery
            else None
        ),
    )