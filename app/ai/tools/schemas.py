"""Validated input schemas for AI-facing business tools."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ToolInputBase(BaseModel):
    """Base configuration for tool inputs."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class OrderStatusInput(ToolInputBase):
    """Input for checking order status."""

    order_id: str = Field(
        min_length=1,
        max_length=50,
    )
    customer_id: UUID


class PaymentStatusInput(ToolInputBase):
    """Input for checking payment status."""

    order_id: str = Field(
        min_length=1,
        max_length=50,
    )
    customer_id: UUID


class CreateTicketInput(ToolInputBase):
    """Input for creating a support ticket."""

    customer_id: UUID

    category: str = Field(
        min_length=1,
        max_length=50,
    )

    description: str = Field(
        min_length=1,
        max_length=2000,
    )

    priority: Literal[
        "low",
        "normal",
        "high",
        "urgent",
    ] = "normal"

    conversation_id: UUID | None = None


class EscalationInput(ToolInputBase):
    """Input for escalating an issue to human support."""

    customer_id: UUID

    reason: str = Field(
        min_length=1,
        max_length=2000,
    )

    conversation_id: UUID | None = None

    priority: Literal[
        "high",
        "urgent",
    ] = "urgent"