"""LangGraph definition for the customer-support agent."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.ai.agent.argument_extractor import ArgumentExtractor
from app.ai.agent.contracts import AgentRoute, Intent
from app.ai.agent.finalize import GuardrailPipeline, TurnTracer
from app.ai.agent.intent_classifier import IntentClassifier
from app.ai.agent.nodes import (
    build_action_planning_node,
    build_argument_preparation_node,
    build_clarification_node,
    build_classification_node,
    build_finalization_node,
    build_mixed_action_execution_node,
    build_mixed_action_preparation_node,
    build_mixed_action_response_node,
    build_multi_tool_execution_node,
    build_multi_tool_preparation_node,
    build_multi_tool_response_node,
    build_rag_node,
    build_routing_node,
    build_tool_node,
    build_tool_response_node,
    build_tool_selection_node,
)
from app.ai.agent.router import AgentRouter
from app.ai.agent.state import ActionType, AgentState
from app.ai.agent.tool_response import ToolResponseGenerator
from app.ai.rag.generator import GroundedResponseGenerator
from app.ai.rag.retriever import Retriever
from app.ai.tools.registry import ToolRegistry

_TOOL_INTENTS = {
    Intent.ORDER_STATUS,
    Intent.PAYMENT_STATUS,
    Intent.SUPPORT_TICKET,
    Intent.HUMAN_ESCALATION,
}


def _route_after_intent(
    state: AgentState,
) -> Literal[
    "rag",
    "tool",
    "multi_tool",
    "mixed_action",
    "ask_clarification",
    "end",
]:
    """Route execution from the actions produced by the planner."""

    planned_actions = list(
        state.get("planned_actions") or []
    )

    # The planner is authoritative for known actions.
    if planned_actions:
        if len(planned_actions) > 1:
            if all(
                action["action_type"] == ActionType.TOOL
                for action in planned_actions
            ):
                return "multi_tool"

            return "mixed_action"

        action = planned_actions[0]

        if action["action_type"] == ActionType.RAG:
            return "rag"

        if action["action_type"] == ActionType.TOOL:
            return "tool"

    # No executable action was produced.
    # Preserve the existing clarification/end behavior.
    route = state.get("route")

    if route == AgentRoute.ASK_CLARIFICATION:
        return "ask_clarification"

    return "end"


def _route_after_argument_preparation(
    state: AgentState,
) -> Literal[
    "execute_tool",
    "ask_clarification",
]:
    """Choose tool execution or clarification."""

    if state.get("missing_fields"):
        return "ask_clarification"

    return "execute_tool"


def _route_after_multi_tool_preparation(
    state: AgentState,
) -> Literal[
    "execute_multi_tool_actions",
    "ask_clarification",
    "end",
]:
    """Choose multi-tool execution, clarification, or finalization."""

    if state.get("missing_fields"):
        return "ask_clarification"

    if state.get("planned_tool_actions"):
        return "execute_multi_tool_actions"

    return "end"


def _route_after_mixed_action_preparation(
    state: AgentState,
) -> Literal[
    "execute_mixed_actions",
    "ask_clarification",
    "end",
]:
    """Choose mixed execution, clarification, or finalization."""

    if state.get("missing_fields"):
        return "ask_clarification"

    if state.get("prepared_actions"):
        return "execute_mixed_actions"

    return "end"

def build_agent_graph(
    classifier: IntentClassifier,
    router: AgentRouter,
    retriever: Retriever,
    generator: GroundedResponseGenerator,
    tool_registry: ToolRegistry,
    argument_extractor: ArgumentExtractor | None = None,
    tool_response_generator: ToolResponseGenerator | None = None,
    tracer: TurnTracer | None = None,
    guardrails: GuardrailPipeline | None = None,
):
    """Build and compile the customer-support agent graph."""

    graph = StateGraph(AgentState)

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    graph.add_node(
        "classify_intent",
        build_classification_node(classifier),
    )

    graph.add_node(
    "plan_actions",
    build_action_planning_node(),
)


    # ---------------------------------------------------------
    # Routing
    # ---------------------------------------------------------

    graph.add_node(
        "route_intent",
        build_routing_node(router),
    )

    # ---------------------------------------------------------
    # RAG
    # ---------------------------------------------------------

    graph.add_node(
        "rag",
        build_rag_node(
            retriever=retriever,
            generator=generator,
        ),
    )

    # ---------------------------------------------------------
    # Single-tool workflow
    # ---------------------------------------------------------

    graph.add_node(
        "select_tool",
        build_tool_selection_node(),
    )

    graph.add_node(
        "prepare_arguments",
        build_argument_preparation_node(
            argument_extractor
        ),
    )

    graph.add_node(
        "execute_tool",
        build_tool_node(tool_registry),
    )

    # ---------------------------------------------------------
    # Multi-tool workflow
    # ---------------------------------------------------------

    graph.add_node(
        "prepare_multi_tool_actions",
        build_multi_tool_preparation_node(
            argument_extractor
        ),
    )

    graph.add_node(
        "execute_multi_tool_actions",
        build_multi_tool_execution_node(
            tool_registry
        ),
    )

    # ---------------------------------------------------------
    # Mixed RAG + tool workflow
    # ---------------------------------------------------------

    graph.add_node(
        "prepare_mixed_actions",
        build_mixed_action_preparation_node(
            argument_extractor
        ),
    )

    graph.add_node(
        "execute_mixed_actions",
        build_mixed_action_execution_node(
            retriever=retriever,
            generator=generator,
            registry=tool_registry,
        ),
    )

    # ---------------------------------------------------------
    # Clarification
    # ---------------------------------------------------------

    graph.add_node(
        "ask_clarification",
        build_clarification_node(),
    )

    # ---------------------------------------------------------
    # Tool response generation
    # ---------------------------------------------------------

    if tool_response_generator is not None:
        graph.add_node(
            "tool_response",
            build_tool_response_node(
                tool_response_generator
            ),
        )

        graph.add_node(
            "multi_tool_response",
            build_multi_tool_response_node(
                tool_response_generator
            ),
        )

        graph.add_node(
        "mixed_action_response",
        build_mixed_action_response_node(
            tool_response_generator
        ),
    )

    # ---------------------------------------------------------
    # Finalization
    # ---------------------------------------------------------

    graph.add_node(
        "finalize_turn",
        build_finalization_node(
            tracer=tracer,
            guardrails=guardrails,
        ),
    )

    # =========================================================
    # EDGES
    # =========================================================

    # START → classification
    graph.add_edge(
        START,
        "classify_intent",
    )

    # classification → routing
    graph.add_edge(
    "classify_intent",
    "plan_actions",
)

    graph.add_edge(
        "plan_actions",
        "route_intent",
    )
    # ---------------------------------------------------------
    # Route after intent classification
    # ---------------------------------------------------------

    graph.add_conditional_edges(
    "route_intent",
    _route_after_intent,
    {
        "rag": "rag",
        "tool": "select_tool",
        "multi_tool": "prepare_multi_tool_actions",
        "mixed_action": "prepare_mixed_actions",
        "ask_clarification": "ask_clarification",
        "end": "finalize_turn",
    },
)

    # ---------------------------------------------------------
    # RAG
    # ---------------------------------------------------------

    graph.add_edge(
        "rag",
        "finalize_turn",
    )

    # ---------------------------------------------------------
    # Single-tool workflow
    # ---------------------------------------------------------

    graph.add_edge(
        "select_tool",
        "prepare_arguments",
    )

    graph.add_conditional_edges(
        "prepare_arguments",
        _route_after_argument_preparation,
        {
            "execute_tool": "execute_tool",
            "ask_clarification": "ask_clarification",
        },
    )

    # ---------------------------------------------------------
    # Multi-tool preparation
    # ---------------------------------------------------------

    graph.add_conditional_edges(
        "prepare_multi_tool_actions",
        _route_after_multi_tool_preparation,
        {
            "execute_multi_tool_actions": (
                "execute_multi_tool_actions"
            ),
            "ask_clarification": "ask_clarification",
            "end": "finalize_turn",
        },
    )


    # ---------------------------------------------------------
    # Mixed action preparation
    # ---------------------------------------------------------

    graph.add_conditional_edges(
        "prepare_mixed_actions",
        _route_after_mixed_action_preparation,
        {
            "execute_mixed_actions": "execute_mixed_actions",
            "ask_clarification": "ask_clarification",
            "end": "finalize_turn",
        },
    )


    # ---------------------------------------------------------
    # Single-tool execution → response
    # ---------------------------------------------------------

    if tool_response_generator is not None:
        graph.add_edge(
            "execute_tool",
            "tool_response",
        )

        graph.add_edge(
            "tool_response",
            "finalize_turn",
        )

        # -----------------------------------------------------
        # Multi-tool execution → response
        # -----------------------------------------------------

        graph.add_edge(
            "execute_multi_tool_actions",
            "multi_tool_response",
        )

        graph.add_edge(
        "execute_mixed_actions",
        "mixed_action_response",
        )

        graph.add_edge(
            "mixed_action_response",
            "finalize_turn",
        )

        graph.add_edge(
            "multi_tool_response",
            "finalize_turn",
        )

    else:
        graph.add_edge(
            "execute_tool",
            "finalize_turn",
        )

        graph.add_edge(
            "execute_multi_tool_actions",
            "finalize_turn",
        )

        graph.add_edge(
        "execute_mixed_actions",
        "finalize_turn",
    )

    # ---------------------------------------------------------
    # Clarification
    # ---------------------------------------------------------

    graph.add_edge(
        "ask_clarification",
        "finalize_turn",
    )

    # ---------------------------------------------------------
    # Finalization → END
    # ---------------------------------------------------------

    graph.add_edge(
        "finalize_turn",
        END,
    )

    return graph.compile()