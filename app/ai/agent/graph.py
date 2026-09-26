"""LangGraph definition for the customer-support agent."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.ai.agent.argument_extractor import ArgumentExtractor
from app.ai.agent.contracts import AgentRoute
from app.ai.agent.intent_classifier import IntentClassifier
from app.ai.agent.nodes import (
    build_argument_preparation_node,
    build_clarification_node,
    build_classification_node,
    build_rag_node,
    build_routing_node,
    build_tool_node,
    build_tool_response_node,
    build_tool_selection_node,
)
from app.ai.agent.router import AgentRouter
from app.ai.agent.state import AgentState
from app.ai.agent.tool_response import ToolResponseGenerator
from app.ai.rag.generator import GroundedResponseGenerator
from app.ai.rag.retriever import Retriever
from app.ai.tools.registry import ToolRegistry


def _route_after_intent(
    state: AgentState,
) -> Literal["rag", "tool", "end"]:
    """Select the next workflow based on the classified route."""
    route = state.get("route")

    if route == AgentRoute.RAG:
        return "rag"

    if route in {
        AgentRoute.TOOL,
        AgentRoute.TICKET,
        AgentRoute.ESCALATE,
    }:
        return "tool"

    return "end"


def build_agent_graph(
    classifier: IntentClassifier,
    router: AgentRouter,
    retriever: Retriever,
    generator: GroundedResponseGenerator,
    tool_registry: ToolRegistry,
    argument_extractor: ArgumentExtractor | None = None,
    tool_response_generator: ToolResponseGenerator | None = None,
):
    """Build and compile the customer-support agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node(
        "classify_intent",
        build_classification_node(classifier),
    )

    graph.add_node(
        "route_intent",
        build_routing_node(router),
    )

    graph.add_node(
        "rag",
        build_rag_node(
            retriever=retriever,
            generator=generator,
        ),
    )

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
        "ask_clarification",
        build_clarification_node(),
    )

    graph.add_node(
        "execute_tool",
        build_tool_node(tool_registry),
    )

    if tool_response_generator is not None:
        graph.add_node(
            "tool_response",
            build_tool_response_node(tool_response_generator),
        )

    graph.add_edge(
        START,
        "classify_intent",
    )

    graph.add_edge(
        "classify_intent",
        "route_intent",
    )

    graph.add_conditional_edges(
        "route_intent",
        _route_after_intent,
        {
            "rag": "rag",
            "tool": "select_tool",
            "end": END,
        },
    )

    graph.add_edge(
        "rag",
        END,
    )

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

    if tool_response_generator is not None:
        graph.add_edge(
            "execute_tool",
            "tool_response",
        )

        graph.add_edge(
            "tool_response",
            END,
        )
    else:
        graph.add_edge(
            "execute_tool",
            END,
        )

    graph.add_edge(
        "ask_clarification",
        END,
    )

    return graph.compile()

def _route_after_argument_preparation(
    state: AgentState,
) -> Literal["execute_tool", "ask_clarification"]:
    """Choose tool execution or clarification."""
    if state.get("missing_fields"):
        return "ask_clarification"

    return "execute_tool"