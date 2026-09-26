"""Deterministic routing from classified intent to agent workflow."""

from __future__ import annotations

from app.ai.agent.contracts import AgentRoute, Intent


class AgentRoutingError(RuntimeError):
    """Raised when an intent cannot be routed safely."""


_INTENT_TO_ROUTE: dict[Intent, AgentRoute] = {
    Intent.KNOWLEDGE_QUERY: AgentRoute.RAG,
    Intent.ORDER_STATUS: AgentRoute.TOOL,
    Intent.PAYMENT_STATUS: AgentRoute.TOOL,
    Intent.REFUND: AgentRoute.RAG,
    Intent.CANCELLATION: AgentRoute.RAG,
    Intent.SUPPORT_TICKET: AgentRoute.TICKET,
    Intent.HUMAN_ESCALATION: AgentRoute.ESCALATE,
    Intent.UNKNOWN: AgentRoute.ASK_CLARIFICATION,
}


class AgentRouter:
    """Map a validated intent to a deterministic workflow route."""

    def route(self, intent: Intent) -> AgentRoute:
        """Return the route for the supplied intent."""
        try:
            return _INTENT_TO_ROUTE[intent]
        except KeyError as exc:
            raise AgentRoutingError(
                f"Unsupported intent: {intent}"
            ) from exc