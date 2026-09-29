from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ConversationSummaryResponse(BaseModel):
    """Conversation metadata shown in the sidebar."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    title: str = Field(min_length=1, max_length=60)
    updated_at: datetime


class ConversationMessageResponse(BaseModel):
    """A persisted conversation message."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1, max_length=20)
    content: str = Field(min_length=1)
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    """Complete persisted conversation."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    messages: list[ConversationMessageResponse]