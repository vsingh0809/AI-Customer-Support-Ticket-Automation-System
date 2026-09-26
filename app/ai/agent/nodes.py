"""LangGraph nodes for agent orchestration."""

from __future__ import annotations

from collections.abc import Callable

from app.ai.agent.argument_extractor import (
    ArgumentExtractionError,
    ArgumentExtractor,
)
from app.ai.agent.contracts import AgentRoute, Intent
from app.ai.agent.intent_classifier import (
    IntentClassificationError,
    IntentClassifier,
)
from app.ai.agent.router import AgentRouter, AgentRoutingError
from app.ai.agent.state import AgentState
from app.ai.agent.tool_response import (
    ToolResponseGenerationError,
    ToolResponseGenerator,
    ToolResult,
)
from app.ai.rag.generator import (
    GenerationError,
    GroundedResponseGenerator,
)
from app.ai.rag.retriever import RetrievalError, Retriever
from app.ai.tools.registry import ToolRegistry, ToolRegistryError


def build_classification_node(
    classifier: IntentClassifier,
) -> Callable[[AgentState], dict]:
    """Build a node that classifies the current customer message."""

    def classify_intent(state: AgentState) -> dict:
        message = state.get("user_message", "")

        if not message.strip():
            return {
                "errors": [
                    "Customer message cannot be empty",
                ]
            }

        try:
            result = classifier.classify(message)
        except IntentClassificationError as exc:
            return {
                "errors": [str(exc)],
                "intent": None,
            }

        return {
            "intent": result.intent,
        }

    return classify_intent


def build_routing_node(
    router: AgentRouter,
) -> Callable[[AgentState], dict]:
    """Build a node that maps intent to a workflow route."""

    def route_intent(state: AgentState) -> dict:
        intent = state.get("intent")

        if intent is None:
            return {
                "errors": [
                    "Cannot route without a classified intent",
                ]
            }

        try:
            route = router.route(intent)
        except AgentRoutingError as exc:
            return {
                "errors": [str(exc)],
                "route": None,
            }

        return {
            "route": route,
        }

    return route_intent


def build_rag_node(
    retriever: Retriever,
    generator: GroundedResponseGenerator,
) -> Callable[[AgentState], dict]:
    """Build a node that performs retrieval and grounded generation."""

    def run_rag(state: AgentState) -> dict:
        query = state.get("user_message", "")

        if not query.strip():
            return {
                "errors": [
                    *state.get("errors", []),
                    "Cannot run RAG without a customer message",
                ],
                "retrieved_context": [],
                "sources": [],
                "response": None,
            }

        try:
            results = retriever.retrieve(query)
        except RetrievalError as exc:
            return {
                "errors": [
                    *state.get("errors", []),
                    str(exc),
                ],
                "retrieved_context": [],
                "sources": [],
                "response": None,
            }

        try:
            response = generator.generate(
                query=query,
                results=results,
            )
        except GenerationError as exc:
            return {
                "errors": [
                    *state.get("errors", []),
                    str(exc),
                ],
                "retrieved_context": results,
                "sources": [],
                "response": None,
            }

        return {
            "retrieved_context": results,
            "sources": list(response.sources),
            "response": response.answer,
        }

    return run_rag

def build_clarification_node() -> Callable[[AgentState], dict]:
    """Build a node that returns a clarification question."""

    def ask_clarification(state: AgentState) -> dict:
        question = state.get("clarification_question")

        if not question:
            question = "Please provide the missing information."

        return {
            "response": question,
        }

    return ask_clarification

def build_tool_selection_node() -> Callable[[AgentState], dict]:
    """Select the business tool required for the current intent."""

    def select_tool(state: AgentState) -> dict:
        intent = state.get("intent")

        tool_by_intent = {
            Intent.ORDER_STATUS: "check_order_status",
            Intent.PAYMENT_STATUS: "check_payment_status",
            Intent.SUPPORT_TICKET: "create_support_ticket",
            Intent.HUMAN_ESCALATION: "escalate_to_human",
        }

        tool_name = tool_by_intent.get(intent)

        if tool_name is None:
            return {
                "errors": [
                    *state.get("errors", []),
                    f"No business tool mapped for intent: {intent}",
                ],
                "tool_name": None,
            }

        return {
            "tool_name": tool_name,
        }

    return select_tool

def build_tool_node(
    registry: ToolRegistry,
) -> Callable[[AgentState], dict]:
    """Build a node that executes the selected business tool."""

    def execute_tool(state: AgentState) -> dict:
        tool_name = state.get("tool_name")

        if not tool_name:
            return {
                "errors": [
                    *state.get("errors", []),
                    "No tool selected",
                ],
                "tool_result": None,
            }

        tool_arguments = dict(
            state.get("tool_arguments", {})
        )

        customer_id = state.get("customer_id")

        if customer_id is not None:
            tool_arguments["customer_id"] = customer_id

        if tool_name in {
            "create_support_ticket",
            "escalate_to_human",
        }:
            conversation_id = state.get("conversation_id")

            if conversation_id is not None:
                tool_arguments["conversation_id"] = conversation_id

        try:
            result = registry.execute(
                tool_name,
                tool_arguments,
            )
        except ToolRegistryError as exc:
            return {
                "errors": [
                    *state.get("errors", []),
                    str(exc),
                ],
                "tool_result": None,
            }

        return {
            "tool_result": result.model_dump(),
        }

    return execute_tool


def build_argument_preparation_node(
    extractor: ArgumentExtractor,
) -> Callable[[AgentState], dict]:
    """Build a node that prepares arguments for a selected tool."""

    def prepare_arguments(state: AgentState) -> dict:
        intent = state.get("intent")
        message = state.get("user_message", "")

        if intent is None:
            return {
                "errors": [
                    *state.get("errors", []),
                    "Cannot prepare tool arguments without intent",
                ],
                "tool_arguments": {},
            }

        # Human escalation does not require additional information.
        # The customer's message itself provides the escalation reason.
        if intent == Intent.HUMAN_ESCALATION:
            return {
                "tool_arguments": {
                    "reason": message.strip(),
                    "priority": "urgent",
                },
                "missing_fields": [],
                "clarification_question": None,
                "route": AgentRoute.ESCALATE,
            }

        try:
            result = extractor.prepare(
                intent=intent,
                message=message,
            )
        except ArgumentExtractionError as exc:
            return {
                "errors": [
                    *state.get("errors", []),
                    str(exc),
                ],
                "tool_arguments": {},
                "missing_fields": [],
                "clarification_question": None,
            }

        route = state.get("route")

        if result.missing_fields:
            route = AgentRoute.ASK_CLARIFICATION

        return {
            "tool_arguments": result.arguments,
            "missing_fields": list(result.missing_fields),
            "clarification_question": result.clarification_question,
            "route": route,
        }

    return prepare_arguments


def build_tool_response_node(
    generator: ToolResponseGenerator,
) -> Callable[[AgentState], dict]:
    """Build a node that converts a tool result into a customer response."""

    def generate_tool_response(state: AgentState) -> dict:
        tool_result_data = state.get("tool_result")

        if not tool_result_data:
            return {
                "errors": [
                    *state.get("errors", []),
                    "No tool result available for response generation",
                ],
                "response": None,
            }

        intent = state.get("intent")
        tool_name = state.get("tool_name")
        user_message = state.get("user_message", "")

        if intent is None or not tool_name:
            return {
                "errors": [
                    *state.get("errors", []),
                    "Missing intent or tool name for response generation",
                ],
                "response": None,
            }

        try:
            tool_result = ToolResult.model_validate(
                tool_result_data
            )

            result = generator.generate(
                user_message=user_message,
                intent=intent,
                tool_name=tool_name,
                tool_result=tool_result,
            )
        except ToolResponseGenerationError as exc:
            return {
                "errors": [
                    *state.get("errors", []),
                    str(exc),
                ],
                "response": None,
            }

        return {
            "response": result.answer,
        }

    return generate_tool_response
