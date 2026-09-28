"""Schemas for support-ticket APIs."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

TicketPriority = Literal["low", "normal", "high", "urgent"]


class CreateTicketRequest(BaseModel):
    """Request body for creating a support ticket."""

    model_config = ConfigDict(extra="forbid")

    customer_id: UUID
    category: str = Field(min_length=1, max_length=50)
    description: str = Field(min_length=1, max_length=2000)
    priority: TicketPriority = "normal"
    conversation_id: UUID | None = None


class TicketResponse(BaseModel):
    """Public support-ticket representation."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    customer_id: UUID
    category: str
    description: str
    priority: str
    status: str
    conversation_id: UUID | None