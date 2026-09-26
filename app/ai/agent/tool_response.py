"""Generate customer-facing responses from business tool results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from app.ai.agent.contracts import Intent
from app.ai.tools.base import ToolResult


class ToolResponseGenerationError(Exception):
    """Raised when tool-result response generation fails."""


class ToolResponseLLMProvider(Protocol):
    """LLM interface for rendering tool results."""

    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class ToolResponse:
    """Customer-facing response generated from a tool result."""

    answer: str


class ToolResponseGenerator:
    """Convert authoritative tool results into customer-facing responses."""

    def __init__(self, provider: ToolResponseLLMProvider):
        self._provider = provider

    def generate(
        self,
        *,
        user_message: str,
        intent: Intent,
        tool_name: str,
        tool_result: ToolResult,
    ) -> ToolResponse:
        """Generate a safe customer-facing response."""

        if not tool_result.success:
            return ToolResponse(
                answer=self._failure_message(tool_name)
            )

        prompt = self._build_prompt(
            user_message=user_message,
            intent=intent,
            tool_name=tool_name,
            tool_result=tool_result,
        )

        try:
            answer = self._provider.generate(prompt)
        except Exception as exc:
            raise ToolResponseGenerationError(
                "Failed to generate tool response"
            ) from exc

        if not answer.strip():
            raise ToolResponseGenerationError(
                "Tool response generator returned an empty answer"
            )

        return ToolResponse(answer=answer.strip())

    @staticmethod
    def _failure_message(tool_name: str) -> str:
        """Return a deterministic fallback for tool failure."""

        messages = {
            "check_order_status": (
                "I couldn't retrieve your order status right now. "
                "Please try again or contact support."
            ),
            "check_payment_status": (
                "I couldn't retrieve your payment status right now. "
                "Please try again or contact support."
            ),
            "create_support_ticket": (
                "I couldn't create your support ticket right now. "
                "Please try again or contact support."
            ),
            "escalate_to_human": (
                "I couldn't complete the escalation right now. "
                "Please try again or contact support."
            ),
        }

        return messages.get(
            tool_name,
            "I couldn't complete that request right now. Please try again."
        )

    @staticmethod
    def _build_prompt(
        *,
        user_message: str,
        intent: Intent,
        tool_name: str,
        tool_result: ToolResult,
    ) -> str:
        """Build a prompt that restricts the LLM to tool-provided facts."""

        data = json.dumps(
            tool_result.data,
            default=str,
            ensure_ascii=False,
        )

        return f"""
You are a customer support response generator.

Respond to the customer's message using ONLY the authoritative
information contained in the tool result.

Do not invent:
- order information
- payment information
- ticket IDs
- dates
- statuses
- amounts
- explanations not present in the tool result

Keep the response concise, clear, and customer-friendly.

Customer message:
{user_message}

Intent:
{intent.value}

Tool:
{tool_name}

Authoritative tool result:
{data}

Return only the customer-facing response.
""".strip()