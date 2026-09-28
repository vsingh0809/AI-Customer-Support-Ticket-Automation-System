"""LLM-based intent classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ai.agent.contracts import Intent
from app.ai.agent.state import ConversationTurn
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
    intents: tuple[Intent, ...] = Field(default=())

    @model_validator(mode="after")
    def validate_intents(self) -> IntentClassification:
        """Keep the legacy primary intent aligned with detected intents."""
        if not self.intents:
            return self.model_copy(
                update={"intents": (self.intent,)}
            )

        if self.intents[0] != self.intent:
            raise ValueError(
                "Primary intent must match the first detected intent"
            )

        return self


class IntentClassifier:
    """Classify a customer message into a supported intent."""

    def __init__(
        self,
        provider: StructuredLLMProvider,
    ) -> None:
        self._provider = provider

    def classify(self,
                  message: str,
                  conversation_history: Sequence[ConversationTurn] |None= None
                  ) -> IntentClassification:
        """Classify one customer message."""
        if not message.strip():
            raise IntentClassificationError(
                "Customer message cannot be empty"
            )

        try:
            result = self._provider.generate_json(
                system_prompt=_build_system_prompt(),
                user_prompt=_build_user_prompt(message=message.strip(),
                                               conversation_history=conversation_history)
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

Classify the customer's message into one or more supported intents.

A message may contain multiple independent customer requests.

Examples:

- "Where is my order 45821?"
  → ["order_status"]

- "Was my payment successful for order 45821?"
  → ["payment_status"]

- "Check my order 45821 and tell me whether the payment was successful."
  → ["order_status", "payment_status"]

Rules:
1. Return JSON only.
2. Return at least one supported intent.
3. Use only the supported intent values.
4. Do not invent intent values.
5. If multiple requests are present, include every applicable intent.
6. Preserve the logical order of the requested actions.
7. The first intent is the primary intent.
8. Do not include explanations.
9. Conversation history may be provided as context.
10. Use history only to resolve references or follow-up requests.
11. The current customer message has priority over older history.
12. Never let an older turn override an explicit current request.

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
5. Conversation history may be provided as context.
6. Use history only to resolve references or follow-up requests.
7. The current customer message has priority over older history.
8. Never let an older turn override an explicit current request.

Required JSON format:
{
  "intent": "primary_supported_intent",
  "intents": [
    "supported_intent_1",
    "supported_intent_2"
  ]
}
"""

def _build_user_prompt(
    *,
    message: str,
    conversation_history: Sequence[ConversationTurn] | None,
) -> str:
    """Build the classification prompt with optional conversation context."""
    history = list(conversation_history or [])

    if not history:
        return message

    current_turn_already_present = (
        history[-1]["role"] == "user"
        and history[-1]["content"] == message
    )

    if not current_turn_already_present:
        history.append(
            {
                "role": "user",
                "content": message,
            }
        )

    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in history
    )

    return f"""Conversation history:
<conversation_history>
{history_text}
</conversation_history>

Classify the current customer message.

The conversation history is the canonical conversational context.
Use it to resolve follow-up references.

The current customer message has priority over older turns.
"""