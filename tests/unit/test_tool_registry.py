import pytest

from app.ai.tools.base import ToolResult
from app.ai.tools.registry import ToolRegistry, ToolRegistryError


class FakeTool:
    name = "fake_tool"

    description = "Fake tool for testing."

    def execute(self, **kwargs):
        return ToolResult.ok(
            {
                "received": kwargs,
            }
        )


def test_register_and_get_tool() -> None:
    registry = ToolRegistry()
    tool = FakeTool()

    registry.register(tool)

    assert registry.get("fake_tool") is tool


def test_execute_registered_tool() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool())

    result = registry.execute(
        "fake_tool",
        {"value": "hello"},
    )

    assert result.success is True
    assert result.data["received"]["value"] == "hello"


def test_unknown_tool_is_rejected() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolRegistryError,
        match="Tool not found",
    ):
        registry.get("missing_tool")


def test_empty_tool_name_is_rejected() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolRegistryError,
        match="Tool name is required",
    ):
        registry.get("   ")


def test_duplicate_tool_registration_is_rejected() -> None:
    registry = ToolRegistry()

    registry.register(FakeTool())

    with pytest.raises(
        ToolRegistryError,
        match="already registered",
    ):
        registry.register(FakeTool())