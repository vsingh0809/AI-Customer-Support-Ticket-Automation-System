"""Streamlit customer-support chat application."""

from __future__ import annotations

from typing import TypedDict
from uuid import UUID

import streamlit as st

from frontend.streamlit.api_client import (
    ChatAPIClient,
    ChatAPIError,
)
from frontend.streamlit.config import get_settings


class ChatMessage(TypedDict):
    """A message rendered in the customer conversation."""

    role: str
    content: str


def _initialize_session_state(customer_id: UUID) -> None:
    """Initialize persistent Streamlit session state."""

    if "customer_id" not in st.session_state:
        st.session_state.customer_id = customer_id

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


def _start_new_conversation() -> None:
    """Reset the active conversation while preserving identity."""

    st.session_state.conversation_id = None
    st.session_state.messages = []


def _render_empty_state() -> None:
    """Render guidance before the customer sends the first message."""

    st.info(
        "Start a conversation by asking about your order, payment, "
        "refund, cancellation, or support request."
    )


def _render_messages() -> None:
    """Render all locally loaded conversation messages."""

    messages: list[ChatMessage] = st.session_state.messages

    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def _load_conversation_history(
    client: ChatAPIClient,
) -> list:
    """Load conversation summaries for the sidebar."""

    customer_id: UUID = st.session_state.customer_id

    try:
        return client.list_conversations(
            customer_id=customer_id,
        )
    except ChatAPIError as exc:
        st.sidebar.error(str(exc))
        return []


def _select_conversation(
    client: ChatAPIClient,
    conversation_id: UUID,
) -> bool:
    """Load a persisted conversation into the active session."""

    customer_id: UUID = st.session_state.customer_id

    try:
        conversation = client.get_conversation(
            customer_id=customer_id,
            conversation_id=conversation_id,
        )
    except ChatAPIError as exc:
        st.sidebar.error(str(exc))
        return False

    st.session_state.conversation_id = (
        conversation.conversation_id
    )

    st.session_state.messages = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in conversation.messages
        if message.role in {"user", "assistant"}
    ]

    return True


def _render_sidebar(
    client: ChatAPIClient,
) -> None:
    """Render new-chat control and persisted conversation history."""

    with st.sidebar:
        st.subheader("Conversation")

        if st.button(
            "＋ New chat",
            use_container_width=True,
        ):
            _start_new_conversation()
            st.rerun()

        st.divider()

        st.caption("Recent conversations")

        conversations = _load_conversation_history(client)

        if not conversations:
            st.caption("No previous conversations.")

        current_conversation_id = (
            st.session_state.conversation_id
        )

        for conversation in conversations:
            is_active = (
                conversation.conversation_id
                == current_conversation_id
            )

            label = (
                f"● {conversation.title}"
                if is_active
                else conversation.title
            )

            if st.button(label, key=f"conversation-{conversation.conversation_id}", use_container_width=True) and _select_conversation(client, conversation.conversation_id):
                   
                   st.rerun()


        if current_conversation_id is not None:
            st.divider()

            st.caption("Active conversation")

            with st.expander("Technical details"):
                st.code(str(current_conversation_id))


def _send_message(
    client: ChatAPIClient,
    message: str,
) -> bool:
    """Send a customer message to the backend chat workflow."""

    customer_id: UUID = st.session_state.customer_id
    conversation_id: UUID | None = st.session_state.conversation_id

    st.session_state.messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    with st.chat_message("user"):
        st.write(message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = client.chat(
                    customer_id=customer_id,
                    message=message,
                    conversation_id=conversation_id,
                )
            except ChatAPIError as exc:
                st.error(str(exc))
                return False

        st.write(result.response)

    st.session_state.conversation_id = result.conversation_id

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.response,
        }
    )

    return True


def main() -> None:
    """Run the Streamlit customer-support application."""

    st.set_page_config(
        page_title="AI Customer Support",
        page_icon="💬",
        layout="centered",
    )

    settings = get_settings()

    _initialize_session_state(
        settings.demo_customer_id
    )

    client = ChatAPIClient(settings)

    st.title("AI Customer Support")
    st.caption(
        "Ask about orders, payments, policies, refunds, "
        "or request support."
    )

    _render_sidebar(client)

    if not st.session_state.messages:
        _render_empty_state()
    else:
        _render_messages()

    message = st.chat_input(
        "How can we help you?"
    )

    if message:
        success = _send_message(
            client=client,
            message=message,
        )

        if success:
            st.rerun()


if __name__ == "__main__":
    main()