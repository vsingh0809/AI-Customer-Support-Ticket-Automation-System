from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

from app.ai.tools.order_tools import OrderTools
from app.core.exceptions import ApplicationServiceError


def test_check_order_status_success() -> None:
    service = Mock()

    service.get_by_external_id.return_value = SimpleNamespace(
        external_order_id="45821",
        status="shipped",
        total_amount=Decimal("1499.00"),
        expected_delivery=datetime(
            2026,
            9,
            28,
            12,
            0,
            tzinfo=UTC,
        ),
    )

    tool = OrderTools(service)

    result = tool.check_order_status("45821")

    assert result.success is True
    assert result.data["order_id"] == "45821"
    assert result.data["status"] == "shipped"
    assert result.data["total_amount"] == "1499.00"
    assert result.data["expected_delivery"] is not None

    service.get_by_external_id.assert_called_once_with(
        "45821",
        None,
    )


def test_check_order_status_not_found() -> None:
    service = Mock()
    service.get_by_external_id.return_value = None

    tool = OrderTools(service)

    result = tool.check_order_status("99999")

    assert result.success is False
    assert result.error_code == "ORDER_NOT_FOUND"

    service.get_by_external_id.assert_called_once_with(
        "99999",
        None,
    )


def test_check_order_status_rejects_empty_id() -> None:
    service = Mock()

    tool = OrderTools(service)

    result = tool.check_order_status("   ")

    assert result.success is False
    assert result.error_code == "INVALID_ORDER_ID"

    service.get_by_external_id.assert_not_called()


def test_check_order_status_handles_service_failure() -> None:
    service = Mock()

    service.get_by_external_id.side_effect = ApplicationServiceError(
        "database unavailable"
    )

    tool = OrderTools(service)

    result = tool.check_order_status("45821")

    assert result.success is False
    assert result.error_code == "ORDER_LOOKUP_FAILED"

    service.get_by_external_id.assert_called_once_with(
        "45821",
        None,
    )