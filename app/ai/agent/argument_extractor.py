"""Structured extraction of arguments required by business tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.ai.agent.contracts import Intent
from app.ai.agent.state import ConversationTurn


class ArgumentExtractionError(RuntimeError):
    """Raised when tool argument extraction fails."""


class StructuredArgumentProvider(Protocol):
    """Provider contract for structured argument extraction."""

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        """Generate structured JSON."""


class ExtractedToolArguments(BaseModel):
    """Arguments explicitly extracted from the customer message."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    order_id: str | None = Field(
        default=None,
        max_length=50,
    )

    category: str | None = Field(
        default=None,
        max_length=50,
    )

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    reason: str | None = Field(
        default=None,
        max_length=2000,
    )

    priority: str | None = Field(
        default=None,
        max_length=20,
    )


class ArgumentPreparationResult(BaseModel):
    """Validated arguments and missing required fields."""

    model_config = ConfigDict(frozen=True)

    arguments: dict[str, str]
    missing_fields: tuple[str, ...] = ()
    clarification_question: str | None = None


class ArgumentExtractor:
    """Extract and validate tool arguments for the classified intent."""

    def __init__(
        self,
        provider: StructuredArgumentProvider,
    ) -> None:
        self._provider = provider

    def prepare(
        self,
        *,
        intent: Intent,
        message: str,
        conversation_history: Sequence[ConversationTurn] | None = None,

    ) -> ArgumentPreparationResult:
        """Extract arguments and determine whether anything is missing."""
        if not message.strip():
            raise ArgumentExtractionError(
                "Customer message cannot be empty"
            )

        try:
            raw_result = self._provider.generate_json(
                system_prompt=_build_system_prompt(intent),
                user_prompt=_build_user_prompt(
                    message=message.strip(),
                    conversation_history=conversation_history,
                                ),
            )

            extracted = ExtractedToolArguments.model_validate(
                raw_result
            )

        except ArgumentExtractionError:
            raise
        except Exception as exc:
            raise ArgumentExtractionError(
                "Tool argument extraction failed"
            ) from exc

        arguments = _clean_arguments(extracted)
        required_fields = _required_fields(intent)

        missing_fields = tuple(
            field
            for field in required_fields
            if not arguments.get(field)
        )

        clarification_question = None

        if missing_fields:
            clarification_question = _build_clarification_question(
                intent,
                missing_fields,
            )

        return ArgumentPreparationResult(
            arguments=arguments,
            missing_fields=missing_fields,
            clarification_question=clarification_question,
        )


def _required_fields(intent: Intent) -> tuple[str, ...]:
    """Return user-provided arguments required for each intent."""
    if intent in {
        Intent.ORDER_STATUS,
        Intent.PAYMENT_STATUS,
    }:
        return ("order_id",)

    if intent == Intent.SUPPORT_TICKET:
        return ("category", "description")

    if intent == Intent.HUMAN_ESCALATION:
        return ("reason",)

    return ()


def _clean_arguments(
    extracted: ExtractedToolArguments,
) -> dict[str, str]:
    """Return only non-empty extracted values."""
    values = extracted.model_dump()

    return {
        key: value.strip()
        for key, value in values.items()
        if value is not None and value.strip()
    }


def _build_system_prompt(intent: Intent) -> str:
    """Build intent-aware extraction instructions."""
    return f"""You extract customer-provided arguments for a
NovaMart support workflow.

Customer intent:
{intent.value}

Return JSON only.

Extract information explicitly present in the current customer message.

If a required argument is missing from the current message, conversation
history may be used only to resolve a follow-up reference.

The current message has priority over conversation history.

Never invent values.

Never invent:
- order IDs
- ticket categories
- reasons
- priorities
- descriptions

Available fields:
- order_id
- category
- description
- reason
- priority

Use null when a field is not explicitly present.

Required JSON format:
{{
  "order_id": null,
  "category": null,
  "description": null,
  "reason": null,
  "priority": null
}}
"""


def _build_clarification_question(
    intent: Intent,
    missing_fields: tuple[str, ...],
) -> str:
    """Create a deterministic clarification question."""
    if intent in {
        Intent.ORDER_STATUS,
        Intent.PAYMENT_STATUS,
    } and "order_id" in missing_fields:
        return "Please provide your order ID so I can check it."

    if intent == Intent.SUPPORT_TICKET:
        if "category" in missing_fields and "description" in missing_fields:
            return "Please provide the issue category and describe the problem."

        if "category" in missing_fields:
            return "What category best describes your issue?"

        return "Please describe the issue you need support with."

    if intent == Intent.HUMAN_ESCALATION:
        return "Please briefly describe why you need human support."

    return "Please provide the missing information."

def _build_user_prompt(
    *,
    message: str,
    conversation_history: Sequence[ConversationTurn] | None,
) -> str:
    """Build the extraction prompt with optional conversation context."""
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

Extract the arguments required for the current customer request.

Rules:
1. Use the current customer message first.
2. Use conversation history only to resolve follow-up references.
3. An explicit current value overrides older values.
4. Never invent values.
"""