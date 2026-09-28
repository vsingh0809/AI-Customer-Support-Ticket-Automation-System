from unittest.mock import Mock
from uuid import uuid4

from app.ai.agent.argument_extractor import ArgumentExtractor
from app.ai.agent.graph import build_agent_graph
from app.ai.agent.intent_classifier import IntentClassifier
from app.ai.agent.router import AgentRouter
from app.ai.agent.state import ActionType
from app.ai.agent.tool_response import ToolResponseGenerator
from app.ai.rag.generator import (
    GeneratedResponse,
)
from app.ai.rag.retriever import RetrievalError
from app.ai.tools.base import ToolResult
from app.ai.tools.registry import ToolRegistry


class FakeStructuredProvider:
    def __init__(self, intents: list[str]) -> None:
        self.intents = intents

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        return {
            "intent": self.intents[0],
            "intents": self.intents,
        }


class FakeArgumentProvider:
    def __init__(self, order_id: str | None) -> None:
        self.order_id = order_id

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }


class FakeToolResponseProvider:
    def generate(self, prompt: str) -> str:
        return "Your order status information is available."


def _build_argument_extractor(
    order_id: str | None = "45821",
) -> ArgumentExtractor:
    return ArgumentExtractor(
        FakeArgumentProvider(order_id)
    )


def _build_registry(
    tool: Mock,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def _build_graph(
    *,
    intents: list[str],
    order_id: str | None = "45821",
    retriever: Mock,
    generator: Mock,
    tool_registry: ToolRegistry,
):
    classifier = IntentClassifier(
        FakeStructuredProvider(intents)
    )

    return build_agent_graph(
        classifier=classifier,
        router=AgentRouter(),
        retriever=retriever,
        generator=generator,
        tool_registry=tool_registry,
        argument_extractor=_build_argument_extractor(
            order_id
        ),
        tool_response_generator=ToolResponseGenerator(
            FakeToolResponseProvider()
        ),
    )


def test_mixed_tool_then_rag_preserves_action_order() -> None:
    retriever = Mock()
    generator = Mock()

    retriever.retrieve.return_value = ["refund-policy-context"]

    generator.generate.return_value = GeneratedResponse(
        answer="Refunds are available according to policy.",
        sources=(),
    )

    order_tool = Mock()
    order_tool.name = "check_order_status"
    order_tool.execute.return_value = ToolResult.ok(
        {
            "order_id": "45821",
            "status": "shipped",
            "total_amount": "1499.00",
            "expected_delivery": None,
        }
    )

    graph = _build_graph(
        intents=["order_status", "refund"],
        retriever=retriever,
        generator=generator,
        tool_registry=_build_registry(order_tool),
    )

    result = graph.invoke(
        {
            "conversation_id": uuid4(),
            "customer_id": uuid4(),
            "user_message": (
                "Check my order 45821 and tell me "
                "your refund policy."
            ),
            "errors": [],
        }
    )

    assert result["planned_actions"][0]["action_type"] == (
        ActionType.TOOL
    )
    assert result["planned_actions"][1]["action_type"] == (
        ActionType.RAG
    )

    assert result["prepared_actions"][0]["action_type"] == (
        ActionType.TOOL
    )
    assert result["prepared_actions"][1]["action_type"] == (
        ActionType.RAG
    )

    assert result["action_results"][0]["action_type"] == (
        ActionType.TOOL
    )
    assert result["action_results"][1]["action_type"] == (
        ActionType.RAG
    )

    order_tool.execute.assert_called_once()
    retriever.retrieve.assert_called_once()

    assert (
        "Your order status information is available."
        in result["response"]
    )
    assert (
        "Refunds are available according to policy."
        in result["response"]
    )


def test_mixed_rag_then_tool_preserves_action_order() -> None:
    retriever = Mock()
    generator = Mock()

    retriever.retrieve.return_value = ["refund-policy-context"]

    generator.generate.return_value = GeneratedResponse(
        answer="Refunds are available according to policy.",
        sources=(),
    )

    order_tool = Mock()
    order_tool.name = "check_order_status"
    order_tool.execute.return_value = ToolResult.ok(
        {
            "order_id": "45821",
            "status": "shipped",
            "total_amount": "1499.00",
            "expected_delivery": None,
        }
    )

    graph = _build_graph(
        intents=["refund", "order_status"],
        retriever=retriever,
        generator=generator,
        tool_registry=_build_registry(order_tool),
    )

    result = graph.invoke(
        {
            "conversation_id": uuid4(),
            "customer_id": uuid4(),
            "user_message": (
                "Tell me your refund policy and "
                "check order 45821."
            ),
            "errors": [],
        }
    )

    assert result["action_results"][0]["action_type"] == (
        ActionType.RAG
    )
    assert result["action_results"][1]["action_type"] == (
        ActionType.TOOL
    )

    assert (
        result["response"].index(
            "Refunds are available according to policy."
        )
        < result["response"].index(
            "Your order status information is available."
        )
    )


def test_mixed_action_asks_for_missing_tool_argument() -> None:
    retriever = Mock()
    generator = Mock()
    order_tool = Mock()
    order_tool.name = "check_order_status"

    graph = _build_graph(
        intents=["refund", "order_status"],
        order_id=None,
        retriever=retriever,
        generator=generator,
        tool_registry=_build_registry(order_tool),
    )

    result = graph.invoke(
        {
            "conversation_id": uuid4(),
            "customer_id": uuid4(),
            "user_message": (
                "Tell me your refund policy and "
                "check my order."
            ),
            "errors": [],
        }
    )

    assert result["missing_fields"] == [
        "order_status.order_id",
    ]

    assert result["response"]

    retriever.retrieve.assert_not_called()
    order_tool.execute.assert_not_called()


def test_mixed_action_keeps_success_when_rag_fails() -> None:
    retriever = Mock()
    generator = Mock()

    retriever.retrieve.side_effect = RetrievalError(
        "Knowledge-base retrieval failed"
    )

    order_tool = Mock()
    order_tool.name = "check_order_status"
    order_tool.execute.return_value = ToolResult.ok(
        {
            "order_id": "45821",
            "status": "shipped",
            "total_amount": "1499.00",
            "expected_delivery": None,
        }
    )

    graph = _build_graph(
        intents=["refund", "order_status"],
        retriever=retriever,
        generator=generator,
        tool_registry=_build_registry(order_tool),
    )

    result = graph.invoke(
        {
            "conversation_id": uuid4(),
            "customer_id": uuid4(),
            "user_message": (
                "Tell me your refund policy and "
                "check order 45821."
            ),
            "errors": [],
        }
    )

    assert len(result["action_results"]) == 2

    assert result["action_results"][0]["success"] is False
    assert result["action_results"][1]["success"] is True

    assert "Knowledge-base retrieval failed" in result["errors"]

    assert result["response"]

    order_tool.execute.assert_called_once()

def test_mixed_action_keeps_success_when_tool_fails() -> None:
    retriever = Mock()
    generator = Mock()

    retriever.retrieve.return_value = ["refund-policy-context"]

    generator.generate.return_value = GeneratedResponse(
        answer="Refunds are available according to policy.",
        sources=(),
    )

    order_tool = Mock()
    order_tool.name = "check_order_status"
    order_tool.execute.return_value = ToolResult.failure(
        error_code="ORDER_LOOKUP_FAILED",
        error_message="Order service is temporarily unavailable.",
    )

    graph = _build_graph(
        intents=["refund", "order_status"],
        retriever=retriever,
        generator=generator,
        tool_registry=_build_registry(order_tool),
    )

    result = graph.invoke(
        {
            "conversation_id": uuid4(),
            "customer_id": uuid4(),
            "user_message": (
                "Tell me your refund policy and "
                "check order 45821."
            ),
            "errors": [],
        }
    )

    assert len(result["action_results"]) == 2

    assert result["action_results"][0]["success"] is True
    assert result["action_results"][0]["action_type"] == (
        ActionType.RAG
    )

    assert result["action_results"][1]["success"] is False
    assert result["action_results"][1]["action_type"] == (
        ActionType.TOOL
    )

    assert result["action_results"][1]["tool_name"] == (
        "check_order_status"
    )

    assert result["response"]

    assert (
        "Refunds are available according to policy."
        in result["response"]
    )

    assert (
        "I couldn't retrieve your order status right now."
        in result["response"]
    )

    order_tool.execute.assert_called_once()
    retriever.retrieve.assert_called_once()    