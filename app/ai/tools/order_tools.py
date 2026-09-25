"""Order-related tools exposed to the AI agent."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import ValidationError

from app.ai.tools.base import ToolResult
from app.ai.tools.schemas import OrderStatusInput
from app.core.exceptions import ApplicationServiceError
from app.services.order_service import OrderService


class OrderTools:
    """Business tools for order-related customer requests."""

    tool_name = "check_order_status"

    description = (
        "Check the current status, total amount, and expected delivery "
        "date of a customer's order."
    )

    def __init__(self, order_service: OrderService) -> None:
        self._order_service = order_service

    def check_order_status(
        self,
        order_id: str,
        customer_id: UUID | None = None,
    ) -> ToolResult:
        """Return the current status of an order."""
        if not order_id or not order_id.strip():
            return ToolResult.failure(
                error_code="INVALID_ORDER_ID",
                error_message="Order ID is required.",
            )

        normalized_order_id = order_id.strip()

        try:
            order = self._order_service.get_by_external_id(
                normalized_order_id,
                customer_id,
            )
        except ApplicationServiceError:
            return ToolResult.failure(
                error_code="ORDER_LOOKUP_FAILED",
                error_message="Unable to retrieve the order right now.",
            )

        if order is None:
            return ToolResult.failure(
                error_code="ORDER_NOT_FOUND",
                error_message=(
                    f"Order {normalized_order_id} was not found."
                ),
            )

        return ToolResult.ok(
            {
                "order_id": order.external_order_id
                if hasattr(order, "external_order_id")
                else normalized_order_id,
                "status": order.status,
                "total_amount": str(order.total_amount),
                "expected_delivery": (
                    order.expected_delivery.isoformat()
                    if order.expected_delivery
                    else None
                ),
            }
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Validate and execute the order-status tool."""
        try:
            params = OrderStatusInput.model_validate(kwargs)
        except ValidationError:
            return ToolResult.failure(
                error_code="INVALID_TOOL_PARAMETERS",
                error_message="Invalid parameters for check_order_status.",
            )

        return self.check_order_status(
            order_id=params.order_id,
            customer_id=params.customer_id,
        )