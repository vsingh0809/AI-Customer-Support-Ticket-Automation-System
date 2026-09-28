"""Schemas for payment APIs."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class PaymentResponse(BaseModel):
    """Public payment representation."""

    model_config = ConfigDict(from_attributes=True)

    payment_id: UUID
    order_id: str
    transaction_id: str
    status: str
    amount: str
    created_at: str | None