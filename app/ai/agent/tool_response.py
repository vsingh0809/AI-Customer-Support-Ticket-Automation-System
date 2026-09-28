"""Generate customer-facing responses from business tool results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.ai.agent.contracts import Intent
from app.ai.tools.base import ToolResult


class ToolResponseGenerationError(Exception):
    """Raised when tool-result response generation cannot be completed."""


class ToolResponseLLMProvider(Protocol):
    """LLM interface for rendering tool results."""

    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class ToolResponse:
    """Customer-facing response generated from tool results."""

    answer: str


@dataclass(frozen=True)
class ToolActionResult:
    """One tool result available for combined response generation."""

    intent: Intent
    tool_name: str
    tool_result: ToolResult


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
        """Generate a safe customer-facing response for one tool result."""

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

    def generate_multiple(
        self,
        *,
        user_message: str,
        results: Sequence[ToolActionResult],
    ) -> ToolResponse:
        """
        Generate one coherent response from multiple authoritative
        tool results.

        The LLM is preferred for natural-language rendering. If that
        rendering fails, deterministic rendering is used so successful
        business-tool results are preserved and the API does not return
        a null assistant response solely because the response LLM failed.
        """
        if not results:
            raise ToolResponseGenerationError(
                "No tool results provided"
            )

        successful_results = [
            item for item in results if item.tool_result.success
        ]

        failed_results = [
            item for item in results if not item.tool_result.success
        ]

        # If every tool failed, avoid unnecessary LLM generation
        # and return deterministic fallbacks.
        if not successful_results:
            answers = [
                self._failure_message(item.tool_name)
                for item in failed_results
            ]

            return ToolResponse(
                answer="\n\n".join(answers)
            )

        prompt = self._build_multi_prompt(
            user_message=user_message,
            results=results,
        )

        try:
            answer = self._provider.generate(prompt)
        except Exception:
            return self._build_multi_fallback(results)

        if not answer.strip():
            return self._build_multi_fallback(results)

        return ToolResponse(answer=answer.strip())

    @classmethod
    def _build_multi_fallback(
        cls,
        results: Sequence[ToolActionResult],
    ) -> ToolResponse:
        """Render multiple tool results without using the response LLM."""

        answers = [
            cls._result_fallback_message(item)
            for item in results
        ]

        answer = "\n\n".join(
            message for message in answers if message
        ).strip()

        if not answer:
            raise ToolResponseGenerationError(
                "Unable to build a deterministic multi-tool response"
            )

        return ToolResponse(answer=answer)

    @classmethod
    def _result_fallback_message(
        cls,
        item: ToolActionResult,
    ) -> str:
        """Render one tool result deterministically."""

        if not item.tool_result.success:
            return cls._failure_message(item.tool_name)

        data = item.tool_result.data or {}

        if item.tool_name == "check_order_status":
            return cls._order_success_message(data)

        if item.tool_name == "check_payment_status":
            return cls._payment_success_message(data)

        if item.tool_name == "create_support_ticket":
            return cls._ticket_success_message(data)

        if item.tool_name == "escalate_to_human":
            return cls._escalation_success_message(data)

        return "The requested operation was completed successfully."

    @staticmethod
    def _order_success_message(data: dict[str, object]) -> str:
        """Render an order-status result using only returned fields."""

        order_id = data.get("order_id")
        status = data.get("status")
        total_amount = data.get("total_amount")
        expected_delivery = data.get("expected_delivery")

        subject = (
            f"Order #{order_id}"
            if order_id is not None
            else "Your order"
        )

        parts: list[str] = []

        if status is not None:
            parts.append(f"{subject} is currently {status}.")
        else:
            parts.append(f"{subject} was found successfully.")

        if total_amount is not None:
            parts.append(f"The order total is {total_amount}.")

        if expected_delivery is not None:
            parts.append(
                f"The expected delivery date is {expected_delivery}."
            )
        else:
            parts.append(
                "An expected delivery date is not currently available."
            )

        return " ".join(parts)

    @staticmethod
    def _payment_success_message(data: dict[str, object]) -> str:
        """Render a payment-status result using only returned fields."""

        order_id = data.get("order_id")
        status = data.get("status")
        amount = data.get("amount")
        payment_id = data.get("payment_id")
        transaction_id = data.get("transaction_id")

        subject = (
            f"Payment for order #{order_id}"
            if order_id is not None
            else "Your payment"
        )

        parts: list[str] = []

        if status is not None:
            parts.append(f"{subject} is {status}.")
        else:
            parts.append(f"{subject} was found successfully.")

        if amount is not None:
            parts.append(f"The payment amount is {amount}.")

        if payment_id is not None:
            parts.append(f"Payment ID: {payment_id}.")

        if transaction_id is not None:
            parts.append(f"Transaction ID: {transaction_id}.")

        return " ".join(parts)

    @staticmethod
    def _ticket_success_message(data: dict[str, object]) -> str:
        """Render a support-ticket creation result safely."""

        ticket_id = data.get("ticket_id")
        status = data.get("status")

        if ticket_id is not None and status is not None:
            return (
                f"Your support ticket #{ticket_id} was created successfully "
                f"and is currently {status}."
            )

        if ticket_id is not None:
            return (
                f"Your support ticket #{ticket_id} was created successfully."
            )

        if status is not None:
            return (
                "Your support ticket was created successfully. "
                f"Its current status is {status}."
            )

        return "Your support ticket was created successfully."

    @staticmethod
    def _escalation_success_message(data: dict[str, object]) -> str:
        """Render a human-escalation result safely."""

        ticket_id = data.get("ticket_id")
        status = data.get("status")

        if ticket_id is not None and status is not None:
            return (
                "Your request has been escalated to human support. "
                f"Escalation ticket #{ticket_id} is currently {status}."
            )

        if ticket_id is not None:
            return (
                "Your request has been escalated to human support. "
                f"Escalation ticket #{ticket_id} was created."
            )

        if status is not None:
            return (
                "Your request has been escalated to human support. "
                f"The escalation status is {status}."
            )

        return "Your request has been escalated to human support."

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
            "I couldn't complete that request right now. Please try again.",
        )

    @staticmethod
    def _build_prompt(
        *,
        user_message: str,
        intent: Intent,
        tool_name: str,
        tool_result: ToolResult,
    ) -> str:
        """Build a prompt for one authoritative tool result."""

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

    @staticmethod
    def _build_multi_prompt(
        *,
        user_message: str,
        results: Sequence[ToolActionResult],
    ) -> str:
        """Build one prompt containing every authoritative tool result."""

        tool_sections: list[str] = []

        for index, item in enumerate(results, start=1):
            data = json.dumps(
                item.tool_result.data,
                default=str,
                ensure_ascii=False,
            )

            tool_sections.append(
                f"""
Action {index}
Intent:
{item.intent.value}

Tool:
{item.tool_name}

Success:
{item.tool_result.success}

Authoritative result:
{data}
""".strip()
            )

        combined_results = "\n\n".join(tool_sections)

        return f"""
You are a customer support response generator.

The customer made one message containing multiple requests.

Produce ONE coherent customer-facing response that addresses
every request using ONLY the authoritative results supplied below.

Rules:
- Address every successful action.
- Never claim an action succeeded unless its authoritative result
  says success=true.
- Never invent information.
- Never infer missing order information.
- Never infer missing payment information.
- Never invent dates, statuses, amounts, IDs, or explanations.
- If one action failed, clearly state that the specific action
  could not be completed.
- Do not say information is unavailable when another supplied
  authoritative result explicitly contains that information.
- Do not mention tools, internal workflows, JSON, prompts,
  or implementation details.
- Keep the response concise and logically organized.

Customer message:
{user_message}

Authoritative action results:
{combined_results}

Return only the final customer-facing response.
""".strip()
