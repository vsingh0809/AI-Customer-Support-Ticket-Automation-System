from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from app.ai.agent.argument_extractor import ArgumentExtractor
from app.ai.agent.contracts import Intent
from app.ai.agent.graph import build_agent_graph
from app.ai.agent.intent_classifier import IntentClassifier
from app.ai.agent.router import AgentRouter
from app.ai.agent.state import ActionType
from app.ai.tools.base import ToolResult


class FakeStructuredProvider:
    def __init__(self, intents: list[str]) -> None:
        self._intents = intents

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        return {
            "intent": self._intents[0],
            "intents": self._intents,
        }


class FakeArgumentProvider:
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        return {
            "order_id": "45821",
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }


def _base_state(message: str) -> dict:
    return {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": message,
        "errors": [],
    }


def test_graph_populates_planned_actions_before_tool_execution() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider(
            [
                "order_status",
                "payment_status",
            ]
        )
    )

    extractor = ArgumentExtractor(
        FakeArgumentProvider()
    )

    registry = Mock()

    registry.execute.side_effect = [
        ToolResult.ok(
            {
                "order_id": "45821",
                "status": "shipped",
            }
        ),
        ToolResult.ok(
            {
                "order_id": "45821",
                "status": "paid",
            }
        ),
    ]

    response_generator = Mock()

    response_generator.generate_multiple.return_value = (
        SimpleNamespace(
            answer="Order and payment checked successfully."
        )
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=AgentRouter(),
        retriever=Mock(),
        generator=Mock(),
        tool_registry=registry,
        argument_extractor=extractor,
        tool_response_generator=response_generator,
    )

    result = graph.invoke(
        _base_state(
            "Please check my order 45821 and tell me "
            "whether the payment was successful."
        )
    )

    assert result["planned_actions"] == [
        {
            "intent": Intent.ORDER_STATUS,
            "action_type": ActionType.TOOL,
            "tool_name": "check_order_status",
        },
        {
            "intent": Intent.PAYMENT_STATUS,
            "action_type": ActionType.TOOL,
            "tool_name": "check_payment_status",
        },
    ]

    assert registry.execute.call_count == 2
    assert result["response"] == (
        "Order and payment checked successfully."
    )


def test_graph_plans_rag_action_for_refund_request() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider(["refund"])
    )

    retriever = Mock()

    rag_response = SimpleNamespace(
        answer="Refund policy response.",
        sources=[],
    )

    generator = Mock()
    generator.generate.return_value = rag_response

    graph = build_agent_graph(
        classifier=classifier,
        router=AgentRouter(),
        retriever=retriever,
        generator=generator,
        tool_registry=Mock(),
        argument_extractor=Mock(),
    )

    result = graph.invoke(
        _base_state(
            "What is your refund policy?"
        )
    )

    assert result["planned_actions"] == [
        {
            "intent": Intent.REFUND,
            "action_type": ActionType.RAG,
            "tool_name": None,
        }
    ]

    generator.generate.assert_called_once()