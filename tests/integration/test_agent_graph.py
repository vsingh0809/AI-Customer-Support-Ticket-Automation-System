from unittest.mock import Mock
from uuid import uuid4

from app.ai.agent.argument_extractor import ArgumentExtractor
from app.ai.agent.contracts import AgentRoute, Intent
from app.ai.agent.graph import build_agent_graph
from app.ai.agent.intent_classifier import IntentClassifier
from app.ai.agent.router import AgentRouter
from app.ai.agent.tool_response import ToolResponseGenerator
from app.ai.rag.generator import (
    GeneratedResponse,
    GenerationError,
)
from app.ai.rag.retriever import RetrievalError
from app.ai.tools.base import ToolResult
from app.ai.tools.registry import ToolRegistry


class FakeStructuredProvider:
    def __init__(self, intent: str) -> None:
        self.intent = intent

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        return {
            "intent": self.intent,
        }

class FakeArgumentProvider:
    def __init__(self, response):
        self.response = response

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ):
        return self.response

class FakeToolResponseProvider:
    """Fake LLM provider for tool-response integration tests."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response

def _build_argument_extractor(response=None) -> ArgumentExtractor:
    return ArgumentExtractor(
        FakeArgumentProvider(
            response
            or {
                "order_id": "45821",
                "category": None,
                "description": None,
                "reason": None,
                "priority": None,
            }
        )
    )   

def _build_tool_response_generator(
    response: str = "Here is the requested information.",
) -> ToolResponseGenerator:
    return ToolResponseGenerator(
        FakeToolResponseProvider(response)
    )

def _build_graph(intent: str):
    classifier = IntentClassifier(
        FakeStructuredProvider(intent)
    )

    router = AgentRouter()

    retriever = Mock()
    generator = Mock()

    generator.generate.return_value = GeneratedResponse(
        answer="Mock grounded response.",
        sources=(),
    )

    tool_registry = ToolRegistry()

    escalation_tool = Mock()
    escalation_tool.name = "escalate_to_human"
    escalation_tool.execute.return_value = ToolResult.ok(
        {
            "escalated": True,
        }
    )

    tool_registry.register(escalation_tool)

    return build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=tool_registry,
        argument_extractor=_build_argument_extractor(),
        tool_response_generator=_build_tool_response_generator(),
    )

def _build_argument_extractor(response=None) -> ArgumentExtractor:
    return ArgumentExtractor(
        FakeArgumentProvider(
            response
            or {
                "order_id": None,
                "category": None,
                "description": None,
                "reason": None,
                "priority": None,
            }
        )
    )

def _build_tool_registry() -> ToolRegistry:
    return ToolRegistry()


def test_graph_routes_knowledge_request_to_rag() -> None:
    graph = _build_graph("knowledge_query")

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "What is your refund policy?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["intent"] == Intent.KNOWLEDGE_QUERY
    assert result["route"] == AgentRoute.RAG
    assert result["errors"] == []


def test_graph_routes_human_request_to_escalation() -> None:
    graph = _build_graph("human_escalation")

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "I want to speak to a human.",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["intent"] == Intent.HUMAN_ESCALATION
    assert result["route"] == AgentRoute.ESCALATE
    assert result["errors"] == []


def test_graph_handles_invalid_classification() -> None:
    graph = _build_graph("invalid_intent")

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "I need help.",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["intent"] is None
    assert result.get("route") is None
    assert result["errors"]


def test_graph_preserves_existing_state() -> None:
    graph = _build_graph("order_status")

    conversation_id = uuid4()
    customer_id = uuid4()

    state = {
        "conversation_id": conversation_id,
        "customer_id": customer_id,
        "user_message": "Where is my order?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["conversation_id"] == conversation_id
    assert result["customer_id"] == customer_id
    assert result["user_message"] == "Where is my order?"

def test_graph_executes_rag_route() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider("knowledge_query")
    )

    router = AgentRouter()

    retriever = Mock()
    generator = Mock()

    retriever.retrieve.return_value = []

    generator.generate.return_value = GeneratedResponse(
        answer="Refunds are available within 7 days.",
        sources=(),
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=_build_tool_registry(),
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "What is your refund policy?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["intent"] == Intent.KNOWLEDGE_QUERY
    assert result["route"] == AgentRoute.RAG
    assert result["response"] == (
        "Refunds are available within 7 days."
    )

    retriever.retrieve.assert_called_once_with(
        "What is your refund policy?"
    )

    generator.generate.assert_called_once()


def test_graph_stores_retrieved_context_and_sources() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider("knowledge_query")
    )

    router = AgentRouter()

    retriever = Mock()
    generator = Mock()

    result_chunk = Mock()

    retriever.retrieve.return_value = [
        result_chunk,
    ]

    generator.generate.return_value = GeneratedResponse(
        answer="Grounded answer.",
        sources=(),
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=_build_tool_registry(),
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "What is your refund policy?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["retrieved_context"] == [result_chunk]
    assert result["sources"] == []
    assert result["response"] == "Grounded answer."

def test_graph_handles_retrieval_failure() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider("knowledge_query")
    )

    router = AgentRouter()

    retriever = Mock()

    retriever.retrieve.side_effect = RetrievalError(
        "Knowledge-base retrieval failed"
    )

    generator = Mock()

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=_build_tool_registry(),
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "What is your refund policy?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["response"] is None
    assert "Knowledge-base retrieval failed" in result["errors"]

    generator.generate.assert_not_called()


def test_graph_handles_generation_failure() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider("knowledge_query")
    )

    router = AgentRouter()

    retriever = Mock()
    generator = Mock()

    retriever.retrieve.return_value = []

    generator.generate.side_effect = GenerationError(
        "Grounded response generation failed"
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=_build_tool_registry(),
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "What is your refund policy?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["response"] is None
    assert "Grounded response generation failed" in result["errors"]


def test_graph_asks_for_missing_order_id() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider("order_status")
    )

    router = AgentRouter()

    retriever = Mock()
    generator = Mock()
    tool_registry = ToolRegistry()

    extractor = _build_argument_extractor(
        {
            "order_id": None,
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=tool_registry,
        argument_extractor=extractor,
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "Where is my order?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["intent"] == Intent.ORDER_STATUS
    assert result["route"] == AgentRoute.ASK_CLARIFICATION

    assert result["missing_fields"] == [
        "order_id",
    ]

    assert result["response"] == (
        "Please provide your order ID so I can check it."
    )

    assert result["tool_arguments"] == {}


def test_clarification_does_not_execute_tool() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider("order_status")
    )

    router = AgentRouter()

    retriever = Mock()
    generator = Mock()
    tool_registry = Mock()

    extractor = _build_argument_extractor(
        {
            "order_id": None,
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=tool_registry,
        argument_extractor=extractor,
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "Where is my order?",
        "errors": [],
    }

    graph.invoke(state)

    tool_registry.execute.assert_not_called()


def test_argument_preparation_allows_complete_order_request() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider("order_status")
    )

    router = AgentRouter()

    retriever = Mock()
    generator = Mock()
    tool_registry = Mock()

    extractor = _build_argument_extractor(
        {
            "order_id": "45821",
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=tool_registry,
        argument_extractor=extractor,
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "Where is my order 45821?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["intent"] == Intent.ORDER_STATUS
    assert result["route"] == AgentRoute.TOOL
    assert result["tool_arguments"] == {
        "order_id": "45821",
    }    

def test_graph_generates_response_from_successful_order_tool() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider("order_status")
    )

    router = AgentRouter()
    retriever = Mock()
    generator = Mock()

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

    tool_registry = ToolRegistry()
    tool_registry.register(order_tool)

    extractor = _build_argument_extractor(
        {
            "order_id": "45821",
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }
    )

    tool_response_provider = FakeToolResponseProvider(
        "Your order 45821 is currently shipped."
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=tool_registry,
        argument_extractor=extractor,
        tool_response_generator=ToolResponseGenerator(
            tool_response_provider
        ),
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "Where is my order 45821?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["tool_name"] == "check_order_status"
    assert result["tool_result"]["success"] is True
    assert result["response"] == (
        "Your order 45821 is currently shipped."
    )

    assert len(tool_response_provider.prompts) == 1

def test_graph_generates_response_from_successful_payment_tool() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider("payment_status")
    )

    router = AgentRouter()
    retriever = Mock()
    generator = Mock()

    payment_tool = Mock()
    payment_tool.name = "check_payment_status"
    payment_tool.execute.return_value = ToolResult.ok(
        {
            "order_id": "45821",
            "payment_id": "pay-45821",
            "transaction_id": "txn-45821",
            "status": "successful",
            "amount": "1499.00",
        }
    )

    tool_registry = ToolRegistry()
    tool_registry.register(payment_tool)

    extractor = _build_argument_extractor(
        {
            "order_id": "45821",
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }
    )

    tool_response_provider = FakeToolResponseProvider(
        "The payment for order 45821 was successful."
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=tool_registry,
        argument_extractor=extractor,
        tool_response_generator=ToolResponseGenerator(
            tool_response_provider
        ),
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "What happened to my payment for order 45821?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["tool_name"] == "check_payment_status"
    assert result["tool_result"]["success"] is True
    assert result["response"] == (
        "The payment for order 45821 was successful."
    )

    assert len(tool_response_provider.prompts) == 1       

def test_graph_generates_fallback_for_failed_tool() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider("order_status")
    )

    router = AgentRouter()
    retriever = Mock()
    generator = Mock()

    order_tool = Mock()
    order_tool.name = "check_order_status"
    order_tool.execute.return_value = ToolResult.failure(
        error_code="Order lookup failed.",
        error_message="some tool failure occurs.",
    )

    tool_registry = ToolRegistry()
    tool_registry.register(order_tool)

    extractor = _build_argument_extractor(
        {
            "order_id": "45821",
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }
    )

    tool_response_provider = FakeToolResponseProvider(
        "This response must not be used."
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=retriever,
        generator=generator,
        tool_registry=tool_registry,
        argument_extractor=extractor,
        tool_response_generator=ToolResponseGenerator(
            tool_response_provider
        ),
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": "Where is my order 45821?",
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["tool_result"]["success"] is False
    assert result["response"] == (
        "I couldn't retrieve your order status right now. "
        "Please try again or contact support."
    )

    assert tool_response_provider.prompts == []