"""LLM-based intent classification."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from app.ai.agent.contracts import Intent
from app.ai.rag.generator import GenerationError


class IntentClassificationError(RuntimeError):
    """Raised when intent classification fails."""


class StructuredLLMProvider(Protocol):
    """Provider contract for structured JSON generation."""

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        """Generate a structured JSON object."""


class IntentClassification(BaseModel):
    """Validated intent classification result."""

    model_config = ConfigDict(frozen=True)

    intent: Intent


class IntentClassifier:
    """Classify a customer message into a supported intent."""

    def __init__(
        self,
        provider: StructuredLLMProvider,
    ) -> None:
        self._provider = provider

    def classify(self, message: str) -> IntentClassification:
        """Classify one customer message."""
        if not message.strip():
            raise IntentClassificationError(
                "Customer message cannot be empty"
            )

        try:
            result = self._provider.generate_json(
                system_prompt=_build_system_prompt(),
                user_prompt=message.strip(),
            )
        except GenerationError as exc:
            raise IntentClassificationError(
                "Intent classification failed"
            ) from exc
        except Exception as exc:
            raise IntentClassificationError(
                "Unexpected intent classification failure"
            ) from exc

        try:
            return IntentClassification.model_validate(result)
        except Exception as exc:
            raise IntentClassificationError(
                "Model returned an invalid intent"
            ) from exc


def _build_system_prompt() -> str:
    """Build the intent-classification instructions."""
    return """You are the intent classifier for NovaMart customer support.

Classify the customer's message into exactly one supported intent.

Supported intents:

- knowledge_query:
  Questions about company information, policies, FAQs, products, shipping,
  payments, refunds, cancellation, or account policies.

- order_status:
  Questions about the status or expected delivery of a specific order.

- payment_status:
  Questions about whether a payment succeeded, failed, was deducted,
  or is associated with an order.

- refund:
  Requests or questions specifically about getting a refund.

- cancellation:
  Requests or questions specifically about cancelling an order.

- support_ticket:
  The customer wants to raise a support issue or create a support ticket.

- human_escalation:
  The customer explicitly requests a human or asks to speak with support staff.

- unknown:
  The request does not clearly match any supported intent.

Rules:
1. Return JSON only.
2. Use exactly one of the supported intent values.
3. Do not invent additional intent values.
4. Do not include explanations.

Required JSON format:
{
  "intent": "one_supported_intent"
}
"""