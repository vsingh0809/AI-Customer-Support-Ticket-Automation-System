"""Contracts used by the AI agent orchestration layer."""

from enum import StrEnum


class Intent(StrEnum):
    """Supported customer intents."""

    KNOWLEDGE_QUERY = "knowledge_query"
    ORDER_STATUS = "order_status"
    PAYMENT_STATUS = "payment_status"
    REFUND = "refund"
    CANCELLATION = "cancellation"
    SUPPORT_TICKET = "support_ticket"
    HUMAN_ESCALATION = "human_escalation"
    UNKNOWN = "unknown"


class AgentRoute(StrEnum):
    """Routes the agent can choose."""

    RAG = "rag"
    TOOL = "tool"
    ASK_CLARIFICATION = "ask_clarification"
    TICKET = "ticket"
    ESCALATE = "escalate"
    RESPOND = "respond"