from unittest.mock import Mock

import pytest

from app.ai.agent.contracts import Intent
from app.ai.agent.tool_response import (
    ToolResponseGenerationError,
    ToolResponseGenerator,
)
from app.ai.tools.base import ToolResult


class FakeToolResponseProvider:
    """Fake LLM provider for deterministic unit tests."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_successful_order_result_generates_customer_response() -> None:
    provider = FakeToolResponseProvider(
        "Your order 45821 is currently shipped."
    )

    generator = ToolResponseGenerator(provider)

    tool_result = ToolResult.ok(
        {
            "order_id": "45821",
            "status": "shipped",
            "total_amount": "1499.00",
            "expected_delivery": None,
        }
    )

    result = generator.generate(
        user_message="Where is my order 45821?",
        intent=Intent.ORDER_STATUS,
        tool_name="check_order_status",
        tool_result=tool_result,
    )

    assert result.answer == (
        "Your order 45821 is currently shipped."
    )

    assert len(provider.prompts) == 1
    assert "45821" in provider.prompts[0]
    assert "shipped" in provider.prompts[0]


def test_successful_payment_result_generates_customer_response() -> None:
    provider = FakeToolResponseProvider(
        "The payment for order 45821 was successful."
    )

    generator = ToolResponseGenerator(provider)

    tool_result = ToolResult.ok(
        {
            "order_id": "45821",
            "payment_id": "pay-45821",
            "transaction_id": "txn-45821",
            "status": "successful",
            "amount": "1499.00",
        }
    )

    result = generator.generate(
        user_message="What happened to my payment for order 45821?",
        intent=Intent.PAYMENT_STATUS,
        tool_name="check_payment_status",
        tool_result=tool_result,
    )

    assert result.answer == (
        "The payment for order 45821 was successful."
    )

    assert len(provider.prompts) == 1
    assert "45821" in provider.prompts[0]
    assert "successful" in provider.prompts[0]


def test_failed_tool_result_returns_deterministic_fallback() -> None:
    provider = Mock()
    generator = ToolResponseGenerator(provider)

    tool_result = ToolResult.failure(
        error_code="Order lookup failed.",
        error_message="some tool failure occures."
    )

    result = generator.generate(
        user_message="Where is my order 45821?",
        intent=Intent.ORDER_STATUS,
        tool_name="check_order_status",
        tool_result=tool_result,
    )

    assert result.answer == (
        "I couldn't retrieve your order status right now. "
        "Please try again or contact support."
    )

    provider.generate.assert_not_called()


def test_empty_llm_response_raises_generation_error() -> None:
    provider = FakeToolResponseProvider("")

    generator = ToolResponseGenerator(provider)

    tool_result = ToolResult.ok(
        {
            "order_id": "45821",
            "status": "shipped",
        }
    )

    with pytest.raises(
        ToolResponseGenerationError,
        match="empty answer",
    ):
        generator.generate(
            user_message="Where is my order 45821?",
            intent=Intent.ORDER_STATUS,
            tool_name="check_order_status",
            tool_result=tool_result,
        )