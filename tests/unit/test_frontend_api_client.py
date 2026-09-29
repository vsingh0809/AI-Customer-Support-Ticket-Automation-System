from unittest.mock import MagicMock, Mock, patch
from uuid import UUID

import httpx2
import pytest

from frontend.streamlit.api_client import (
    ChatAPIClient,
    ChatAPIError,ConversationDetail
)
from frontend.streamlit.config import FrontendSettings
from datetime import UTC, datetime

CUSTOMER_ID = UUID(
    "3f2b8f18-7f3d-4d6e-9a2c-5c3e4a7d8b1f"
)

CONVERSATION_ID = UUID(
    "7b5f2e9c-2f6a-4a8d-9d21-8f1c3e6b7420"
)


def _build_client() -> ChatAPIClient:
    settings = FrontendSettings(
        backend_api_url="http://localhost:8000",
        demo_customer_id=CUSTOMER_ID,
    )

    return ChatAPIClient(settings)


def _build_mock_http_client() -> MagicMock:
    mock_http_client = MagicMock()

    mock_http_client.__enter__.return_value = mock_http_client
    mock_http_client.__exit__.return_value = None

    return mock_http_client


def test_chat_sends_expected_payload() -> None:
    client = _build_client()

    response = Mock()
    response.json.return_value = {
        "conversation_id": str(CONVERSATION_ID),
        "response": "Order #45821 has been shipped.",
    }
    response.raise_for_status.return_value = None

    mock_http_client = _build_mock_http_client()
    mock_http_client.post.return_value = response

    with patch(
        "frontend.streamlit.api_client.httpx2.Client",
        return_value=mock_http_client,
    ):
        result = client.chat(
            customer_id=CUSTOMER_ID,
            message="Where is my order 45821?",
        )

    assert result.conversation_id == CONVERSATION_ID
    assert result.response == (
        "Order #45821 has been shipped."
    )

    mock_http_client.post.assert_called_once_with(
        "/chat",
        json={
            "customer_id": str(CUSTOMER_ID),
            "message": "Where is my order 45821?",
        },
    )


def test_chat_sends_conversation_id_for_follow_up() -> None:
    client = _build_client()

    response = Mock()
    response.json.return_value = {
        "conversation_id": str(CONVERSATION_ID),
        "response": "The order is still in transit.",
    }
    response.raise_for_status.return_value = None

    mock_http_client = _build_mock_http_client()
    mock_http_client.post.return_value = response

    with patch(
        "frontend.streamlit.api_client.httpx2.Client",
        return_value=mock_http_client,
    ):
        result = client.chat(
            customer_id=CUSTOMER_ID,
            conversation_id=CONVERSATION_ID,
            message="When will it arrive?",
        )

    assert result.conversation_id == CONVERSATION_ID

    mock_http_client.post.assert_called_once_with(
        "/chat",
        json={
            "customer_id": str(CUSTOMER_ID),
            "conversation_id": str(CONVERSATION_ID),
            "message": "When will it arrive?",
        },
    )


def test_chat_translates_http_error_to_chat_api_error() -> None:
    client = _build_client()

    request = httpx2.Request(
        "POST",
        "http://localhost:8000/chat",
    )

    response = httpx2.Response(
        503,
        request=request,
        json={"detail": "Service unavailable"},
    )

    mock_http_client = _build_mock_http_client()
    mock_http_client.post.return_value = response

    with patch(
        "frontend.streamlit.api_client.httpx2.Client",
        return_value=mock_http_client,
    ):
        with pytest.raises(ChatAPIError) as exc_info:
            client.chat(
                customer_id=CUSTOMER_ID,
                message="Hello",
            )

    assert exc_info.value.status_code == 503
    assert str(exc_info.value) == (
        "The support service returned an error. "
        "Please try again."
    )


def test_chat_translates_network_error_to_chat_api_error() -> None:
    client = _build_client()

    mock_http_client = _build_mock_http_client()

    mock_http_client.post.side_effect = httpx2.ConnectError(
        "connection failed"
    )

    with patch(
        "frontend.streamlit.api_client.httpx2.Client",
        return_value=mock_http_client,
    ):
        with pytest.raises(ChatAPIError) as exc_info:
            client.chat(
                customer_id=CUSTOMER_ID,
                message="Hello",
            )

    assert str(exc_info.value) == (
        "Unable to reach the support service. "
        "Please try again."
    )


def test_chat_rejects_invalid_backend_response() -> None:
    client = _build_client()

    response = Mock()
    response.json.return_value = {
        "conversation_id": str(CONVERSATION_ID),
    }
    response.raise_for_status.return_value = None

    mock_http_client = _build_mock_http_client()
    mock_http_client.post.return_value = response

    with patch(
        "frontend.streamlit.api_client.httpx2.Client",
        return_value=mock_http_client,
    ):
        with pytest.raises(ChatAPIError) as exc_info:
            client.chat(
                customer_id=CUSTOMER_ID,
                message="Hello",
            )

    assert str(exc_info.value) == (
        "The support service returned an invalid response."
    )

def test_list_conversations_sends_customer_id() -> None:
    client = _build_client()

    response = Mock()
    response.json.return_value = [
        {
            "conversation_id": str(CONVERSATION_ID),
            "title": "Where is my order?",
            "updated_at": datetime.now(UTC).isoformat(),
        }
    ]
    response.raise_for_status.return_value = None

    mock_http_client = MagicMock()
    mock_http_client.__enter__.return_value = mock_http_client
    mock_http_client.__exit__.return_value = None
    mock_http_client.get.return_value = response

    with patch(
        "frontend.streamlit.api_client.httpx2.Client",
        return_value=mock_http_client,
    ):
        result = client.list_conversations(
            customer_id=CUSTOMER_ID,
        )

    assert len(result) == 1
    assert result[0].conversation_id == CONVERSATION_ID
    assert result[0].title == "Where is my order?"

    mock_http_client.get.assert_called_once_with(
        "/conversations",
        params={
            "customer_id": str(CUSTOMER_ID),
        },
    )


def test_get_conversation_returns_persisted_messages() -> None:
    client = _build_client()

    response = Mock()
    response.json.return_value = {
        "conversation_id": str(CONVERSATION_ID),
        "messages": [
            {
                "role": "user",
                "content": "Where is my order?",
                "created_at": datetime.now(UTC).isoformat(),
            },
            {
                "role": "assistant",
                "content": "Please provide your order ID.",
                "created_at": datetime.now(UTC).isoformat(),
            },
        ],
    }
    response.raise_for_status.return_value = None

    mock_http_client = MagicMock()
    mock_http_client.__enter__.return_value = mock_http_client
    mock_http_client.__exit__.return_value = None
    mock_http_client.get.return_value = response

    with patch(
        "frontend.streamlit.api_client.httpx2.Client",
        return_value=mock_http_client,
    ):
        result = client.get_conversation(
            customer_id=CUSTOMER_ID,
            conversation_id=CONVERSATION_ID,
        )

    assert isinstance(result, ConversationDetail)
    assert result.conversation_id == CONVERSATION_ID
    assert len(result.messages) == 2
    assert result.messages[0].role == "user"
    assert result.messages[1].role == "assistant"

    mock_http_client.get.assert_called_once_with(
        f"/conversations/{CONVERSATION_ID}",
        params={
            "customer_id": str(CUSTOMER_ID),
        },
    )


def test_get_conversation_translates_not_found() -> None:
    client = _build_client()

    request = httpx2.Request(
        "GET",
        f"http://localhost:8000/conversations/{CONVERSATION_ID}",
    )

    response = httpx2.Response(
        404,
        request=request,
        json={"detail": "Conversation not found"},
    )

    mock_http_client = MagicMock()
    mock_http_client.__enter__.return_value = mock_http_client
    mock_http_client.__exit__.return_value = None
    mock_http_client.get.return_value = response

    with patch(
        "frontend.streamlit.api_client.httpx2.Client",
        return_value=mock_http_client,
    ):
        with pytest.raises(ChatAPIError) as exc_info:
            client.get_conversation(
                customer_id=CUSTOMER_ID,
                conversation_id=CONVERSATION_ID,
            )

    assert exc_info.value.status_code == 404
    assert str(exc_info.value) == "Conversation not found."    