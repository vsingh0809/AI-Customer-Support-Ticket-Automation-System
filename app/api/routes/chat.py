"""Chat API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_application_service
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_application import ChatApplicationService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: Annotated[
        ChatApplicationService,
        Depends(get_chat_application_service),
    ],
) -> ChatResponse:
    conversation_id, response = service.chat(
        customer_id=request.customer_id,
        message=request.message,
        conversation_id=request.conversation_id,
    )

    return ChatResponse(
        conversation_id=conversation_id,
        response=response,
    )