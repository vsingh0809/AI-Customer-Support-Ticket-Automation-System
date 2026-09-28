"""Schemas for human-escalation APIs."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

EscalationPriority = Literal["high", "urgent"]


class EscalationRequest(BaseModel):
    """Request body for human escalation."""

    model_config = ConfigDict(extra="forbid")

    customer_id: UUID
    reason: str = Field(min_length=1, max_length=2000)
    conversation_id: UUID | None = None
    priority: EscalationPriority = "urgent"