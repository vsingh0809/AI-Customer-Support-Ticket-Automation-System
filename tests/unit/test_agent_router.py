import pytest

from app.ai.agent.contracts import AgentRoute, Intent
from app.ai.agent.router import AgentRouter


@pytest.mark.parametrize(
    ("intent", "expected_route"),
    [
        (
            Intent.KNOWLEDGE_QUERY,
            AgentRoute.RAG,
        ),
        (
            Intent.ORDER_STATUS,
            AgentRoute.TOOL,
        ),
        (
            Intent.PAYMENT_STATUS,
            AgentRoute.TOOL,
        ),
        (
            Intent.REFUND,
            AgentRoute.RAG,
        ),
        (
            Intent.CANCELLATION,
            AgentRoute.RAG,
        ),
        (
            Intent.SUPPORT_TICKET,
            AgentRoute.TICKET,
        ),
        (
            Intent.HUMAN_ESCALATION,
            AgentRoute.ESCALATE,
        ),
        (
            Intent.UNKNOWN,
            AgentRoute.ASK_CLARIFICATION,
        ),
    ],
)
def test_intent_routes_to_expected_workflow(
    intent: Intent,
    expected_route: AgentRoute,
) -> None:
    router = AgentRouter()

    assert router.route(intent) == expected_route

def test_every_supported_intent_has_a_route() -> None:
    router = AgentRouter()

    for intent in Intent:
        assert router.route(intent) in AgentRoute    