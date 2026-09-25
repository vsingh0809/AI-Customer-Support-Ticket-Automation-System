"""Payment-related tools exposed to the AI agent."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import ValidationError

from app.ai.tools.base import ToolResult
from app.ai.tools.schemas import PaymentStatusInput
from app.core.exceptions import ApplicationServiceError
from app.services.order_service import OrderService
from app.services.payment_service import PaymentService


class PaymentTools:
    """Business tools for payment-related customer requests."""

    tool_name = "check_payment_status"

    description = (
        "Check the latest payment status and transaction details "
        "for a customer's order."
    )

    def __init__(
        self,
        order_service: OrderService,
        payment_service: PaymentService,
    ) -> None:
        self._order_service = order_service
        self._payment_service = payment_service

    def check_payment_status(
        self,
        order_id: str,
        customer_id: UUID | None = None,
    ) -> ToolResult:
        """
        Return the latest payment for an external order ID.

        The external order ID is resolved through OrderService first so
        callers never need to provide the internal database UUID.
        """
        if not order_id or not order_id.strip():
            return ToolResult.failure(
                error_code="INVALID_PAYMENT_REFERENCE",
                error_message="Order ID is required to check payment status.",
            )

        normalized_order_id = order_id.strip()

        try:
            order = self._order_service.get_by_external_id(
                normalized_order_id,
                customer_id,
            )

            if order is None:
                return ToolResult.failure(
                    error_code="ORDER_NOT_FOUND",
                    error_message=(
                        f"Order {normalized_order_id} was not found."
                    ),
                )

            payment = self._payment_service.get_latest_for_order(
                order.id
            )
        except ApplicationServiceError:
            return ToolResult.failure(
                error_code="PAYMENT_LOOKUP_FAILED",
                error_message=(
                    "Unable to retrieve payment information right now."
                ),
            )

        if payment is None:
            return ToolResult.failure(
                error_code="PAYMENT_NOT_FOUND",
                error_message=(
                    f"No payment record was found for order "
                    f"{normalized_order_id}."
                ),
            )

        return ToolResult.ok(
            {
                "order_id": normalized_order_id,
                "payment_id": str(payment.id),
                "transaction_id": payment.transaction_id,
                "status": payment.status,
                "amount": str(payment.amount),
                "created_at": (
                    payment.created_at.isoformat()
                    if payment.created_at
                    else None
                ),
            }
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Validate and execute the payment-status tool."""
        try:
            params = PaymentStatusInput.model_validate(kwargs)
        except ValidationError:
            return ToolResult.failure(
                error_code="INVALID_TOOL_PARAMETERS",
                error_message="Invalid parameters for check_payment_status.",
            )

        return self.check_payment_status(
            order_id=params.order_id,
            customer_id=params.customer_id,
        )