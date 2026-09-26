import pytest

from app.ai.agent.contracts import Intent
from app.ai.agent.intent_classifier import (
    IntentClassificationError,
    IntentClassifier,
)


class FakeStructuredProvider:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        self.calls.append(
            (
                system_prompt,
                user_prompt,
            )
        )

        return self.response


class FailingProvider:
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        raise RuntimeError("provider unavailable")


def test_classifies_order_status() -> None:
    provider = FakeStructuredProvider(
        {"intent": "order_status"}
    )

    classifier = IntentClassifier(provider)

    result = classifier.classify(
        "Where is my order 45821?"
    )

    assert result.intent == Intent.ORDER_STATUS
    assert len(provider.calls) == 1
    assert provider.calls[0][1] == "Where is my order 45821?"


def test_classifies_knowledge_query() -> None:
    provider = FakeStructuredProvider(
        {"intent": "knowledge_query"}
    )

    classifier = IntentClassifier(provider)

    result = classifier.classify(
        "What is your refund policy?"
    )

    assert result.intent == Intent.KNOWLEDGE_QUERY


def test_classifies_human_escalation() -> None:
    provider = FakeStructuredProvider(
        {"intent": "human_escalation"}
    )

    classifier = IntentClassifier(provider)

    result = classifier.classify(
        "I want to speak to a human."
    )

    assert result.intent == Intent.HUMAN_ESCALATION


def test_rejects_invalid_intent() -> None:
    provider = FakeStructuredProvider(
        {"intent": "make_me_coffee"}
    )

    classifier = IntentClassifier(provider)

    with pytest.raises(
        IntentClassificationError,
        match="invalid intent",
    ):
        classifier.classify("I need help")


def test_rejects_empty_message() -> None:
    provider = FakeStructuredProvider(
        {"intent": "unknown"}
    )

    classifier = IntentClassifier(provider)

    with pytest.raises(
        IntentClassificationError,
        match="cannot be empty",
    ):
        classifier.classify("   ")

    assert provider.calls == []


def test_handles_provider_failure() -> None:
    classifier = IntentClassifier(
        FailingProvider()
    )

    with pytest.raises(
        IntentClassificationError,
        match="Unexpected intent classification failure",
    ):
        classifier.classify(
            "Where is my order?"
        )


def test_unknown_intent_is_valid() -> None:
    provider = FakeStructuredProvider(
        {"intent": "unknown"}
    )

    classifier = IntentClassifier(provider)

    result = classifier.classify(
        "Tell me something unrelated."
    )

    assert result.intent == Intent.UNKNOWN