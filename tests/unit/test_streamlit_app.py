from datetime import UTC, datetime
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID

import streamlit as st

from frontend.streamlit.api_client import (
    ChatAPIClient,
    ChatAPIError,
    ChatResponse,
    ConversationDetail,
)
from frontend.streamlit.app import (
    _initialize_session_state,
    _render_empty_state,
    _select_conversation,
    _send_message,
    _start_new_conversation,
)

CUSTOMER_ID = UUID(
    "3f2b8f18-7f3d-4d6e-9a2c-5c3e4a7d8b1f"
)

CONVERSATION_ID = UUID(
    "7b5f2e9c-2f6a-4a8d-9d21-8f1c3e6b7420"
)

NEW_CONVERSATION_ID = UUID(
    "9c7e4a1d-6b52-4f83-a921-3d8c7e5b2046"
)


def _clear_session_state() -> None:
    st.session_state.clear()


def _patch_streamlit_chat_ui():
    return (
        patch(
            "frontend.streamlit.app.st.chat_message",
            return_value=MagicMock(),
        ),
        patch(
            "frontend.streamlit.app.st.spinner",
            return_value=MagicMock(),
        ),
        patch(
            "frontend.streamlit.app.st.error",
        ),
    )


def test_initialize_session_state() -> None:
    _clear_session_state()

    _initialize_session_state(CUSTOMER_ID)

    assert st.session_state.customer_id == CUSTOMER_ID
    assert st.session_state.conversation_id is None
    assert st.session_state.messages == []


def test_initialize_session_state_does_not_overwrite_existing_state() -> None:
    _clear_session_state()

    st.session_state.customer_id = CUSTOMER_ID
    st.session_state.conversation_id = CONVERSATION_ID
    st.session_state.messages = [
        {
            "role": "user",
            "content": "Where is my order?",
        }
    ]

    _initialize_session_state(CUSTOMER_ID)

    assert st.session_state.customer_id == CUSTOMER_ID
    assert st.session_state.conversation_id == CONVERSATION_ID
    assert st.session_state.messages == [
        {
            "role": "user",
            "content": "Where is my order?",
        }
    ]


def test_start_new_conversation_clears_conversation_state() -> None:
    _clear_session_state()

    st.session_state.customer_id = CUSTOMER_ID
    st.session_state.conversation_id = CONVERSATION_ID
    st.session_state.messages = [
        {
            "role": "user",
            "content": "Where is my order?",
        },
        {
            "role": "assistant",
            "content": "Please provide your order ID.",
        },
    ]

    _start_new_conversation()

    assert st.session_state.customer_id == CUSTOMER_ID
    assert st.session_state.conversation_id is None
    assert st.session_state.messages == []


def test_send_message_stores_conversation_id_and_messages() -> None:
    _clear_session_state()
    _initialize_session_state(CUSTOMER_ID)

    client = Mock(spec=ChatAPIClient)

    client.chat.return_value = ChatResponse(
        conversation_id=CONVERSATION_ID,
        response="Please provide your order ID.",
    )

    (
        chat_message_patch,
        spinner_patch,
        error_patch,
    ) = _patch_streamlit_chat_ui()

    with (
        chat_message_patch,
        spinner_patch,
        error_patch,
    ):
        _send_message(
            client=client,
            message="Where is my order?",
        )

    client.chat.assert_called_once_with(
        customer_id=CUSTOMER_ID,
        message="Where is my order?",
        conversation_id=None,
    )

    assert st.session_state.conversation_id == CONVERSATION_ID

    assert st.session_state.messages == [
        {
            "role": "user",
            "content": "Where is my order?",
        },
        {
            "role": "assistant",
            "content": "Please provide your order ID.",
        },
    ]


def test_send_message_reuses_existing_conversation_id() -> None:
    _clear_session_state()
    _initialize_session_state(CUSTOMER_ID)

    st.session_state.conversation_id = CONVERSATION_ID
    st.session_state.messages = [
        {
            "role": "user",
            "content": "Where is my order?",
        },
        {
            "role": "assistant",
            "content": "Please provide your order ID.",
        },
    ]

    client = Mock(spec=ChatAPIClient)

    client.chat.return_value = ChatResponse(
        conversation_id=NEW_CONVERSATION_ID,
        response="Order #45821 has been shipped.",
    )

    (
        chat_message_patch,
        spinner_patch,
        error_patch,
    ) = _patch_streamlit_chat_ui()

    with (
        chat_message_patch,
        spinner_patch,
        error_patch,
    ):
        _send_message(
            client=client,
            message="45821",
        )

    client.chat.assert_called_once_with(
        customer_id=CUSTOMER_ID,
        message="45821",
        conversation_id=CONVERSATION_ID,
    )

    assert st.session_state.conversation_id == NEW_CONVERSATION_ID

    assert st.session_state.messages == [
        {
            "role": "user",
            "content": "Where is my order?",
        },
        {
            "role": "assistant",
            "content": "Please provide your order ID.",
        },
        {
            "role": "user",
            "content": "45821",
        },
        {
            "role": "assistant",
            "content": "Order #45821 has been shipped.",
        },
    ]


def test_send_message_api_failure_preserves_conversation_state() -> None:
    _clear_session_state()
    _initialize_session_state(CUSTOMER_ID)

    st.session_state.conversation_id = CONVERSATION_ID
    st.session_state.messages = [
        {
            "role": "user",
            "content": "Where is my order?",
        },
        {
            "role": "assistant",
            "content": "Please provide your order ID.",
        },
    ]

    client = Mock(spec=ChatAPIClient)

    client.chat.side_effect = ChatAPIError(
        "Unable to reach the support service. Please try again."
    )

    (
        chat_message_patch,
        spinner_patch,
        error_patch,
    ) = _patch_streamlit_chat_ui()

    with (
        chat_message_patch,
        spinner_patch,
        error_patch as mock_error,
    ):
        _send_message(
            client=client,
            message="45821",
        )

    client.chat.assert_called_once_with(
        customer_id=CUSTOMER_ID,
        message="45821",
        conversation_id=CONVERSATION_ID,
    )

    assert st.session_state.conversation_id == CONVERSATION_ID

    assert st.session_state.messages == [
        {
            "role": "user",
            "content": "Where is my order?",
        },
        {
            "role": "assistant",
            "content": "Please provide your order ID.",
        },
        {
            "role": "user",
            "content": "45821",
        },
    ]

    mock_error.assert_called_once_with(
        "Unable to reach the support service. Please try again."
    )


def test_render_empty_state() -> None:
    _clear_session_state()

    with patch(
        "frontend.streamlit.app.st.info"
    ) as mock_info:
        _render_empty_state()

    mock_info.assert_called_once_with(
        "Start a conversation by asking about your order, payment, "
        "refund, cancellation, or support request."
    )

def test_select_conversation_loads_persisted_messages() -> None:
    _clear_session_state()
    _initialize_session_state(CUSTOMER_ID)

    client = Mock(spec=ChatAPIClient)

    created_at = datetime.now(UTC)

    client.get_conversation.return_value = ConversationDetail(
        conversation_id=CONVERSATION_ID,
        messages=[
            {
                "role": "user",
                "content": "Where is my order?",
                "created_at": created_at,
            },
            {
                "role": "assistant",
                "content": "Please provide your order ID.",
                "created_at": created_at,
            },
        ],
    )

    with patch(
        "frontend.streamlit.app.st.sidebar.error"
    ):
        result = _select_conversation(
            client=client,
            conversation_id=CONVERSATION_ID,
        )

    assert result is True
    assert (
        st.session_state.conversation_id
        == CONVERSATION_ID
    )

    assert st.session_state.messages == [
        {
            "role": "user",
            "content": "Where is my order?",
        },
        {
            "role": "assistant",
            "content": "Please provide your order ID.",
        },
    ]

    client.get_conversation.assert_called_once_with(
        customer_id=CUSTOMER_ID,
        conversation_id=CONVERSATION_ID,
    )


def test_select_conversation_failure_does_not_replace_active_chat() -> None:
    _clear_session_state()
    _initialize_session_state(CUSTOMER_ID)

    st.session_state.conversation_id = CONVERSATION_ID
    st.session_state.messages = [
        {
            "role": "user",
            "content": "Existing message",
        }
    ]

    client = Mock(spec=ChatAPIClient)

    client.get_conversation.side_effect = ChatAPIError(
        "Conversation not found."
    )

    with patch(
        "frontend.streamlit.app.st.sidebar.error"
    ) as mock_error:
        result = _select_conversation(
            client=client,
            conversation_id=NEW_CONVERSATION_ID,
        )

    assert result is False
    assert (
        st.session_state.conversation_id
        == CONVERSATION_ID
    )
    assert st.session_state.messages == [
        {
            "role": "user",
            "content": "Existing message",
        }
    ]

    mock_error.assert_called_once_with(
        "Conversation not found."
    )    