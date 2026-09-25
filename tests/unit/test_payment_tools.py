from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from app.ai.tools.payment_tools import PaymentTools
from app.core.exceptions import ApplicationServiceError


def _build_order() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        external_order_id="45821",
    )


def _build_payment(order_id) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        order_id=order_id,
        transaction_id="txn-45821",
        status="successful",
        amount=Decimal("1499.00"),
        created_at=datetime(
            2026,
            9,
            25,
            10,
            30,
            tzinfo=UTC,
        ),
    )


def test_check_payment_status_success() -> None:
    order_service = Mock()
    payment_service = Mock()

    order = _build_order()
    payment = _build_payment(order.id)

    order_service.get_by_external_id.return_value = order
    payment_service.get_latest_for_order.return_value = payment

    tool = PaymentTools(
        order_service=order_service,
        payment_service=payment_service,
    )

    customer_id = uuid4()

    result = tool.check_payment_status(
        order_id="45821",
        customer_id=customer_id,
    )

    assert result.success is True
    assert result.data["order_id"] == "45821"
    assert result.data["payment_id"] == str(payment.id)
    assert result.data["transaction_id"] == "txn-45821"
    assert result.data["status"] == "successful"
    assert result.data["amount"] == "1499.00"
    assert result.data["created_at"] is not None

    order_service.get_by_external_id.assert_called_once_with(
        "45821",
        customer_id,
    )

    payment_service.get_latest_for_order.assert_called_once_with(
        order.id,
    )


def test_check_payment_status_order_not_found() -> None:
    order_service = Mock()
    payment_service = Mock()

    order_service.get_by_external_id.return_value = None

    tool = PaymentTools(
        order_service=order_service,
        payment_service=payment_service,
    )

    result = tool.check_payment_status("99999")

    assert result.success is False
    assert result.error_code == "ORDER_NOT_FOUND"

    payment_service.get_latest_for_order.assert_not_called()


def test_check_payment_status_payment_not_found() -> None:
    order_service = Mock()
    payment_service = Mock()

    order = _build_order()

    order_service.get_by_external_id.return_value = order
    payment_service.get_latest_for_order.return_value = None

    tool = PaymentTools(
        order_service=order_service,
        payment_service=payment_service,
    )

    result = tool.check_payment_status("45821")

    assert result.success is False
    assert result.error_code == "PAYMENT_NOT_FOUND"

    payment_service.get_latest_for_order.assert_called_once_with(
        order.id,
    )


def test_check_payment_status_rejects_empty_order_id() -> None:
    order_service = Mock()
    payment_service = Mock()

    tool = PaymentTools(
        order_service=order_service,
        payment_service=payment_service,
    )

    result = tool.check_payment_status("   ")

    assert result.success is False
    assert result.error_code == "INVALID_PAYMENT_REFERENCE"

    order_service.get_by_external_id.assert_not_called()
    payment_service.get_latest_for_order.assert_not_called()


def test_check_payment_status_handles_service_failure() -> None:
    order_service = Mock()
    payment_service = Mock()

    order_service.get_by_external_id.side_effect = ApplicationServiceError(
        "database unavailable"
    )

    tool = PaymentTools(
        order_service=order_service,
        payment_service=payment_service,
    )

    result = tool.check_payment_status("45821")

    assert result.success is False
    assert result.error_code == "PAYMENT_LOOKUP_FAILED"

    payment_service.get_latest_for_order.assert_not_called()


def test_check_payment_status_handles_payment_service_failure() -> None:
    order_service = Mock()
    payment_service = Mock()

    order = _build_order()

    order_service.get_by_external_id.return_value = order
    payment_service.get_latest_for_order.side_effect = ApplicationServiceError(
        "payment database unavailable"
    )

    tool = PaymentTools(
        order_service=order_service,
        payment_service=payment_service,
    )

    result = tool.check_payment_status("45821")

    assert result.success is False
    assert result.error_code == "PAYMENT_LOOKUP_FAILED"