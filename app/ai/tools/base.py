"""Common contracts for business tools."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ToolResult(BaseModel):
    """Standard result returned by every business tool."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def ok(cls, data: dict[str, Any] | None = None) -> ToolResult:
        """Create a successful tool result."""
        return cls(
            success=True,
            data=data or {},
        )

    @classmethod
    def failure(
        cls,
        *,
        error_code: str,
        error_message: str,
    ) -> ToolResult:
        """Create a failed tool result."""
        return cls(
            success=False,
            error_code=error_code,
            error_message=error_message,
        )


class BusinessTool(Protocol):
    """Contract implemented by application business tools."""

    name: str

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the business operation."""