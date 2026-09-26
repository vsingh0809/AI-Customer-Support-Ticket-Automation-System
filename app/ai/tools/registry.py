"""Registry for business tools available to the AI agent."""

from __future__ import annotations

from typing import Any

from app.ai.tools.base import BusinessTool, ToolResult


class ToolRegistryError(RuntimeError):
    """Raised when a tool registry operation fails."""


class ToolRegistry:
    """Register and resolve business tools by stable tool name."""

    def __init__(self) -> None:
        self._tools: dict[str, BusinessTool] = {}

    def register(self, tool: BusinessTool) -> None:
        """Register one business tool."""
        name = getattr(tool, "name", "").strip()

        if not name:
            raise ToolRegistryError(
                "Tool name cannot be empty"
            )

        if name in self._tools:
            raise ToolRegistryError(
                f"Tool already registered: {name}"
            )

        self._tools[name] = tool

    def get(self, name: str) -> BusinessTool:
        """Return a registered tool by name."""
        if not name or not name.strip():
            raise ToolRegistryError(
                "Tool name is required"
            )

        try:
            return self._tools[name.strip()]
        except KeyError as exc:
            raise ToolRegistryError(
                f"Tool not found: {name}"
            ) from exc

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a registered tool."""
        tool = self.get(name)
        return tool.execute(**arguments)