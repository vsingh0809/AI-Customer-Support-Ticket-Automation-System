from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from app.ai.agent.argument_extractor import ArgumentExtractor
from app.ai.agent.contracts import Intent
from app.ai.agent.graph import build_agent_graph
from app.ai.agent.intent_classifier import IntentClassifier
from app.ai.agent.router import AgentRouter
from app.ai.agent.tool_response import (
    ToolActionResult,
    ToolResponseGenerator,
)
from app.ai.tools.base import ToolResult


class FakeStructuredProvider:
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        return {
            "intent": "order_status",
            "intents": [
                "order_status",
                "payment_status",
            ],
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


def test_graph_executes_multiple_tools_sequentially() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider()
    )

    router = AgentRouter()

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

    response_generator.generate_multiple.return_value = SimpleNamespace(
        answer=(
            "Order 45821 is shipped.\n"
            "The payment for order 45821 was successful."
        )
    )

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=Mock(),
        generator=Mock(),
        tool_registry=registry,
        argument_extractor=extractor,
        tool_response_generator=response_generator,
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": (
            "Please check my order 45821 and tell me "
            "whether the payment was successful."
        ),
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["intents"] == [
        Intent.ORDER_STATUS,
        Intent.PAYMENT_STATUS,
    ]

    assert registry.execute.call_count == 2
    response_generator.generate_multiple.assert_called_once()
    response_generator.generate.assert_not_called()

    assert result["response"] == (
        "Order 45821 is shipped.\n"
        "The payment for order 45821 was successful."
    )


def test_multi_tool_request_does_not_execute_when_information_is_missing() -> None:
    classifier = IntentClassifier(
        FakeStructuredProvider()
    )

    router = AgentRouter()

    provider = Mock()

    provider.generate_json.return_value = {
        "order_id": None,
        "category": None,
        "description": None,
        "reason": None,
        "priority": None,
    }

    extractor = ArgumentExtractor(provider)

    registry = Mock()

    graph = build_agent_graph(
        classifier=classifier,
        router=router,
        retriever=Mock(),
        generator=Mock(),
        tool_registry=registry,
        argument_extractor=extractor,
    )

    state = {
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_message": (
            "Check my order and tell me whether the payment succeeded."
        ),
        "errors": [],
    }

    result = graph.invoke(state)

    assert result["missing_fields"]
    assert "order_status.order_id" in result["missing_fields"]
    assert "payment_status.order_id" in result["missing_fields"]

    registry.execute.assert_not_called()


def test_generate_multiple_combines_authoritative_results() -> None:
    provider = Mock()

    provider.generate.return_value = (
        "Your order 45821 is currently shipped. "
        "The payment was successful."
    )

    generator = ToolResponseGenerator(provider)

    results = [
        ToolActionResult(
            intent=Intent.ORDER_STATUS,
            tool_name="check_order_status",
            tool_result=ToolResult.ok(
                {
                    "order_id": "45821",
                    "status": "shipped",
                    "total_amount": "1499.00",
                }
            ),
        ),
        ToolActionResult(
            intent=Intent.PAYMENT_STATUS,
            tool_name="check_payment_status",
            tool_result=ToolResult.ok(
                {
                    "order_id": "45821",
                    "status": "paid",
                    "amount": "1499.00",
                }
            ),
        ),
    ]

    result = generator.generate_multiple(
        user_message=(
            "Please check my order 45821 and tell me "
            "whether the payment was successful."
        ),
        results=results,
    )

    assert result.answer == (
        "Your order 45821 is currently shipped. "
        "The payment was successful."
    )

    provider.generate.assert_called_once()

    prompt = provider.generate.call_args.args[0]

    assert "check_order_status" in prompt
    assert "check_payment_status" in prompt
    assert "shipped" in prompt
    assert "paid" in prompt


def test_generate_multiple_handles_order_failure_and_payment_success() -> None:
    provider = Mock()

    provider.generate.return_value = (
        "I couldn't retrieve the order status, "
        "but the payment for order 45821 was paid."
    )

    generator = ToolResponseGenerator(provider)

    results = [
        ToolActionResult(
            intent=Intent.ORDER_STATUS,
            tool_name="check_order_status",
            tool_result=ToolResult.failure(
                error_code="ORDER_LOOKUP_FAILED",
                error_message="Order lookup failed",
            ),
        ),
        ToolActionResult(
            intent=Intent.PAYMENT_STATUS,
            tool_name="check_payment_status",
            tool_result=ToolResult.ok(
                {
                    "order_id": "45821",
                    "status": "paid",
                }
            ),
        ),
    ]

    result = generator.generate_multiple(
        user_message="Check my order and payment for 45821.",
        results=results,
    )

    assert "order" in result.answer.lower()
    assert "payment" in result.answer.lower()
    assert "45821" in result.answer

    provider.generate.assert_called_once()

    prompt = provider.generate.call_args.args[0]

    assert "Success:\nFalse" in prompt
    assert "Success:\nTrue" in prompt
    assert "ORDER_LOOKUP_FAILED" not in prompt


def test_generate_multiple_handles_order_success_and_payment_failure() -> None:
    provider = Mock()

    provider.generate.return_value = (
        "Your order 45821 is currently shipped, "
        "but the payment status could not be retrieved."
    )

    generator = ToolResponseGenerator(provider)

    results = [
        ToolActionResult(
            intent=Intent.ORDER_STATUS,
            tool_name="check_order_status",
            tool_result=ToolResult.ok(
                {
                    "order_id": "45821",
                    "status": "shipped",
                }
            ),
        ),
        ToolActionResult(
            intent=Intent.PAYMENT_STATUS,
            tool_name="check_payment_status",
            tool_result=ToolResult.failure(
                error_code="PAYMENT_LOOKUP_FAILED",
                error_message="Payment lookup failed",
            ),
        ),
    ]

    result = generator.generate_multiple(
        user_message="Check my order and payment for 45821.",
        results=results,
    )

    assert "order" in result.answer.lower()
    assert "payment" in result.answer.lower()
    assert "45821" in result.answer

    provider.generate.assert_called_once()

    prompt = provider.generate.call_args.args[0]

    assert "Success:\nTrue" in prompt
    assert "Success:\nFalse" in prompt
    assert "check_order_status" in prompt
    assert "check_payment_status" in prompt


def test_generate_multiple_uses_deterministic_fallback_when_llm_fails() -> None:
    provider = Mock()

    provider.generate.side_effect = RuntimeError(
        "Response LLM unavailable"
    )

    generator = ToolResponseGenerator(provider)

    results = [
        ToolActionResult(
            intent=Intent.ORDER_STATUS,
            tool_name="check_order_status",
            tool_result=ToolResult.ok(
                {
                    "order_id": "45821",
                    "status": "shipped",
                }
            ),
        ),
        ToolActionResult(
            intent=Intent.PAYMENT_STATUS,
            tool_name="check_payment_status",
            tool_result=ToolResult.ok(
                {
                    "order_id": "45821",
                    "status": "paid",
                }
            ),
        ),
    ]

    result = generator.generate_multiple(
        user_message=(
            "Please check my order 45821 and tell me "
            "whether the payment was successful."
        ),
        results=results,
    )

    assert result.answer
    assert "Order #45821 is currently shipped." in result.answer
    assert "Payment for order #45821 is paid." in result.answer

    provider.generate.assert_called_once()


def test_generate_multiple_uses_fallback_when_all_tools_fail() -> None:
    provider = Mock()

    generator = ToolResponseGenerator(provider)

    results = [
        ToolActionResult(
            intent=Intent.ORDER_STATUS,
            tool_name="check_order_status",
            tool_result=ToolResult.failure(
                error_code="ORDER_LOOKUP_FAILED",
                error_message="Order lookup failed",
            ),
        ),
        ToolActionResult(
            intent=Intent.PAYMENT_STATUS,
            tool_name="check_payment_status",
            tool_result=ToolResult.failure(
                error_code="PAYMENT_LOOKUP_FAILED",
                error_message="Payment lookup failed",
            ),
        ),
    ]

    result = generator.generate_multiple(
        user_message="Check my order and payment.",
        results=results,
    )

    assert "order status" in result.answer.lower()
    assert "payment status" in result.answer.lower()

    provider.generate.assert_not_called()
