"""HTTP client for the customer-facing support API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import httpx2
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)

from frontend.streamlit.config import FrontendSettings


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: UUID
    message: str = Field(min_length=1, max_length=5000)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    response: str = Field(min_length=1)


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    title: str = Field(min_length=1, max_length=60)
    updated_at: datetime


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1, max_length=20)
    content: str = Field(min_length=1)
    created_at: datetime


class ConversationDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    messages: list[ConversationMessage]


class ChatAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class ChatAPIClient:
    _TIMEOUT_SECONDS = 30.0

    def __init__(self, settings: FrontendSettings) -> None:
        self._base_url = settings.backend_api_url.rstrip("/")

    def chat(
        self,
        *,
        customer_id: UUID,
        message: str,
        conversation_id: UUID | None = None,
    ) -> ChatResponse:
        request = ChatRequest(
            customer_id=customer_id,
            message=message,
            conversation_id=conversation_id,
        )

        try:
            with httpx2.Client(
                base_url=self._base_url,
                timeout=self._TIMEOUT_SECONDS,
            ) as client:
                response = client.post(
                    "/chat",
                    json=request.model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                )
                response.raise_for_status()

        except httpx2.HTTPStatusError as exc:
            raise ChatAPIError(
                "The support service returned an error. "
                "Please try again.",
                status_code=exc.response.status_code,
            ) from exc

        except httpx2.RequestError as exc:
            raise ChatAPIError(
                "Unable to reach the support service. "
                "Please try again."
            ) from exc

        try:
            return ChatResponse.model_validate(
                response.json()
            )
        except (ValueError, ValidationError) as exc:
            raise ChatAPIError(
                "The support service returned an invalid response."
            ) from exc

    def list_conversations(
        self,
        *,
        customer_id: UUID,
    ) -> list[ConversationSummary]:
        """Load all persisted conversations for a customer."""

        try:
            with httpx2.Client(
                base_url=self._base_url,
                timeout=self._TIMEOUT_SECONDS,
            ) as client:
                response = client.get(
                    "/conversations",
                    params={
                        "customer_id": str(customer_id),
                    },
                )
                response.raise_for_status()

        except httpx2.HTTPStatusError as exc:
            raise ChatAPIError(
                "Unable to load conversation history. "
                "Please try again.",
                status_code=exc.response.status_code,
            ) from exc

        except httpx2.RequestError as exc:
            raise ChatAPIError(
                "Unable to reach the support service. "
                "Please try again."
            ) from exc

        try:
            adapter = TypeAdapter(list[ConversationSummary])

            return adapter.validate_python(
                response.json()
            )
        except (ValueError, ValidationError) as exc:
            raise ChatAPIError(
                "The support service returned invalid "
                "conversation history."
            ) from exc

    def get_conversation(
        self,
        *,
        customer_id: UUID,
        conversation_id: UUID,
    ) -> ConversationDetail:
        """Load all persisted messages for a conversation."""

        try:
            with httpx2.Client(
                base_url=self._base_url,
                timeout=self._TIMEOUT_SECONDS,
            ) as client:
                response = client.get(
                    f"/conversations/{conversation_id}",
                    params={
                        "customer_id": str(customer_id),
                    },
                )
                response.raise_for_status()

        except httpx2.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ChatAPIError(
                    "Conversation not found.",
                    status_code=404,
                ) from exc

            raise ChatAPIError(
                "Unable to load the conversation. "
                "Please try again.",
                status_code=exc.response.status_code,
            ) from exc

        except httpx2.RequestError as exc:
            raise ChatAPIError(
                "Unable to reach the support service. "
                "Please try again."
            ) from exc

        try:
            return ConversationDetail.model_validate(
                response.json()
            )
        except (ValueError, ValidationError) as exc:
            raise ChatAPIError(
                "The support service returned an invalid "
                "conversation."
            ) from exc