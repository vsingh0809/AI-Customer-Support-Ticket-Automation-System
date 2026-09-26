import pytest

from app.ai.agent.argument_extractor import (
    ArgumentExtractionError,
    ArgumentExtractor,
)
from app.ai.agent.contracts import Intent


class FakeProvider:
    def __init__(self, response):
        self.response = response

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ):
        return self.response


class FailingProvider:
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ):
        raise RuntimeError("provider unavailable")


def test_extracts_order_id() -> None:
    provider = FakeProvider(
        {
            "order_id": "45821",
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }
    )

    extractor = ArgumentExtractor(provider)

    result = extractor.prepare(
        intent=Intent.ORDER_STATUS,
        message="Where is my order 45821?",
    )

    assert result.arguments == {
        "order_id": "45821",
    }

    assert result.missing_fields == ()
    assert result.clarification_question is None


def test_missing_order_id_requires_clarification() -> None:
    provider = FakeProvider(
        {
            "order_id": None,
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }
    )

    extractor = ArgumentExtractor(provider)

    result = extractor.prepare(
        intent=Intent.ORDER_STATUS,
        message="Where is my order?",
    )

    assert result.arguments == {}
    assert result.missing_fields == ("order_id",)

    assert result.clarification_question == (
        "Please provide your order ID so I can check it."
    )


def test_extracts_ticket_arguments() -> None:
    provider = FakeProvider(
        {
            "order_id": None,
            "category": "payment",
            "description": "Payment was deducted but the order failed.",
            "reason": None,
            "priority": "high",
        }
    )

    extractor = ArgumentExtractor(provider)

    result = extractor.prepare(
        intent=Intent.SUPPORT_TICKET,
        message="My payment was deducted but the order failed.",
    )

    assert result.arguments == {
        "category": "payment",
        "description": "Payment was deducted but the order failed.",
        "priority": "high",
    }

    assert result.missing_fields == ()


def test_missing_ticket_description_requires_clarification() -> None:
    provider = FakeProvider(
        {
            "order_id": None,
            "category": "payment",
            "description": None,
            "reason": None,
            "priority": None,
        }
    )

    extractor = ArgumentExtractor(provider)

    result = extractor.prepare(
        intent=Intent.SUPPORT_TICKET,
        message="I need help with a payment issue.",
    )

    assert result.arguments == {
        "category": "payment",
    }

    assert result.missing_fields == ("description",)


def test_extracts_escalation_reason() -> None:
    provider = FakeProvider(
        {
            "order_id": None,
            "category": None,
            "description": None,
            "reason": "The issue requires manual investigation.",
            "priority": None,
        }
    )

    extractor = ArgumentExtractor(provider)

    result = extractor.prepare(
        intent=Intent.HUMAN_ESCALATION,
        message="I need a human because this requires manual investigation.",
    )

    assert result.arguments == {
        "reason": "The issue requires manual investigation.",
    }

    assert result.missing_fields == ()


def test_no_arguments_required_for_knowledge_query() -> None:
    provider = FakeProvider(
        {
            "order_id": None,
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }
    )

    extractor = ArgumentExtractor(provider)

    result = extractor.prepare(
        intent=Intent.KNOWLEDGE_QUERY,
        message="What is your refund policy?",
    )

    assert result.arguments == {}
    assert result.missing_fields == ()


def test_provider_failure_is_wrapped() -> None:
    extractor = ArgumentExtractor(
        FailingProvider()
    )

    with pytest.raises(
        ArgumentExtractionError,
        match="Tool argument extraction failed",
    ):
        extractor.prepare(
            intent=Intent.ORDER_STATUS,
            message="Where is my order 45821?",
        )


def test_empty_message_is_rejected() -> None:
    provider = FakeProvider({})

    extractor = ArgumentExtractor(provider)

    with pytest.raises(
        ArgumentExtractionError,
        match="Customer message cannot be empty",
    ):
        extractor.prepare(
            intent=Intent.ORDER_STATUS,
            message="   ",
        )