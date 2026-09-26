"""End-of-turn finalization hooks for the AI agent."""

from __future__ import annotations

from typing import Protocol

from app.ai.agent.state import AgentState


class TurnTracer(Protocol):
    """Contract for end-of-turn tracing."""

    def on_turn_end(self, state: AgentState) -> None:
        """Observe a completed agent turn."""


class NoOpTracing:
    """No-op tracing implementation."""

    def on_turn_end(self, state: AgentState) -> None:
        """Do nothing."""
        return


class GuardrailPipeline:
    """Guardrail hook for completed agent turns."""

    def check(self, state: AgentState) -> None:
        """Validate or inspect the completed turn."""
        return