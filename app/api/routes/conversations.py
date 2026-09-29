"""Conversation history API routes."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_conversation_memory_service
from app.api.schemas.conversations import (
    ConversationDetailResponse,
    ConversationMessageResponse,
    ConversationSummaryResponse,
)
from app.services.conversation_memory import ConversationMemoryService

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)


@router.get(
    "",
    response_model=list[ConversationSummaryResponse],
)
def list_conversations(
    customer_id: Annotated[
        UUID,
        Query(...),
    ],
    service: Annotated[
        ConversationMemoryService,
        Depends(get_conversation_memory_service),
    ],
) -> list[ConversationSummaryResponse]:
    """List all conversations belonging to a customer."""

    conversations = service.list_conversations(
        customer_id=customer_id,
    )

    return [
        ConversationSummaryResponse(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            updated_at=conversation.updated_at,
        )
        for conversation in conversations
    ]


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
)
def get_conversation(
    conversation_id: UUID,
    customer_id: Annotated[
        UUID,
        Query(...),
    ],
    service: Annotated[
        ConversationMemoryService,
        Depends(get_conversation_memory_service),
    ],
) -> ConversationDetailResponse:
    """Load all persisted messages for an owned conversation."""

    messages = service.load_conversation_messages(
        conversation_id=conversation_id,
        customer_id=customer_id,
    )

    if messages is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return ConversationDetailResponse(
        conversation_id=conversation_id,
        messages=[
            ConversationMessageResponse(
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in messages
        ],
    )