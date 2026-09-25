from app.ai.tools.base import ToolResult


def test_success_result() -> None:
    result = ToolResult.ok(
        {
            "order_id": "45821",
            "status": "shipped",
        }
    )

    assert result.success is True
    assert result.data["order_id"] == "45821"
    assert result.error_code is None
    assert result.error_message is None


def test_success_result_without_data() -> None:
    result = ToolResult.ok()

    assert result.success is True
    assert result.data == {}


def test_failure_result() -> None:
    result = ToolResult.failure(
        error_code="ORDER_NOT_FOUND",
        error_message="Order 45821 was not found.",
    )

    assert result.success is False
    assert result.data == {}
    assert result.error_code == "ORDER_NOT_FOUND"
    assert result.error_message == "Order 45821 was not found."


def test_failure_result_requires_error_details() -> None:
    result = ToolResult.failure(
        error_code="TOOL_ERROR",
        error_message="Tool execution failed.",
    )

    assert result.success is False
    assert result.error_code
    assert result.error_message