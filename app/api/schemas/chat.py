from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    customer_id: UUID
    message: str = Field(min_length=1, max_length=5000)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: UUID
    response: str