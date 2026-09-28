"""Schemas for order APIs."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class OrderResponse(BaseModel):
    """Public order representation."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    external_order_id: str
    customer_id: UUID
    status: str
    total_amount: str
    expected_delivery: str | None