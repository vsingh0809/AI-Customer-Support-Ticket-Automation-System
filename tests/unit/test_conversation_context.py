from app.ai.agent.argument_extractor import ArgumentExtractor
from app.ai.agent.contracts import Intent
from app.ai.agent.intent_classifier import IntentClassifier


class RecordingIntentProvider:
    def __init__(self) -> None:
        self.user_prompts: list[str] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        self.user_prompts.append(user_prompt)

        return {
            "intent": "order_status",
        }


class ContextAwareArgumentProvider:
    def __init__(self) -> None:
        self.user_prompts: list[str] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        self.user_prompts.append(user_prompt)

        if "99123" in user_prompt:
            order_id = "99123"
        elif "45821" in user_prompt:
            order_id = "45821"
        else:
            order_id = None

        return {
            "order_id": order_id,
            "category": None,
            "description": None,
            "reason": None,
            "priority": None,
        }


def test_intent_classifier_receives_conversation_history() -> None:
    provider = RecordingIntentProvider()
    classifier = IntentClassifier(provider)

    result = classifier.classify(
        "When will it arrive?",
        conversation_history=[
            {
                "role": "user",
                "content": "Where is my order 45821?",
            },
            {
                "role": "assistant",
                "content": "Your order is currently shipped.",
            },
        ],
    )

    assert result.intent == Intent.ORDER_STATUS

    prompt = provider.user_prompts[0]

    assert "Where is my order 45821?" in prompt
    assert "Your order is currently shipped." in prompt
    assert "When will it arrive?" in prompt


def test_argument_extractor_resolves_order_from_history() -> None:
    provider = ContextAwareArgumentProvider()
    extractor = ArgumentExtractor(provider)

    result = extractor.prepare(
        intent=Intent.ORDER_STATUS,
        message="When will it arrive?",
        conversation_history=[
            {
                "role": "user",
                "content": "Where is my order 45821?",
            },
            {
                "role": "assistant",
                "content": "Your order is currently shipped.",
            },
        ],
    )

    assert result.arguments == {
        "order_id": "45821",
    }

    assert result.missing_fields == ()


def test_current_order_id_takes_precedence_over_history() -> None:
    provider = ContextAwareArgumentProvider()
    extractor = ArgumentExtractor(provider)

    result = extractor.prepare(
        intent=Intent.ORDER_STATUS,
        message="Check order 99123 instead.",
        conversation_history=[
            {
                "role": "user",
                "content": "Where is my order 45821?",
            },
            {
                "role": "assistant",
                "content": "Your order is currently shipped.",
            },
        ],
    )

    assert result.arguments == {
        "order_id": "99123",
    }