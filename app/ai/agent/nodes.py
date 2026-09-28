"""LangGraph nodes for agent orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from app.ai.agent.argument_extractor import (
    ArgumentExtractionError,
    ArgumentExtractor,
)
from app.ai.agent.contracts import AgentRoute, Intent
from app.ai.agent.finalize import (
    GuardrailPipeline,
    NoOpTracing,
    TurnTracer,
)
from app.ai.agent.intent_classifier import (
    IntentClassificationError,
    IntentClassifier,
)
from app.ai.agent.router import AgentRouter, AgentRoutingError
from app.ai.agent.state import (
    ActionExecutionResult,
    ActionType,
    AgentState,
    ConversationTurn,
    PlannedAction,
    PreparedAction,
)
from app.ai.agent.tool_response import (
    ToolActionResult,
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

_TOOL_BY_INTENT: dict[Intent, str] = {
    Intent.ORDER_STATUS: "check_order_status",
    Intent.PAYMENT_STATUS: "check_payment_status",
    Intent.SUPPORT_TICKET: "create_support_ticket",
    Intent.HUMAN_ESCALATION: "escalate_to_human",
}

_RAG_INTENTS = {
    Intent.KNOWLEDGE_QUERY,
    Intent.REFUND,
    Intent.CANCELLATION,
}

def build_action_planning_node() -> Callable[[AgentState], dict]:
    """Build a deterministic planner from classified intents."""

    def plan_actions(state: AgentState) -> dict:
        intents = list(state.get("intents") or [])

        if not intents:
            intent = state.get("intent")

            if intent is not None:
                intents = [intent]

        if not intents:
            return {
                "planned_actions": [],
                "errors": [
                    *state.get("errors", []),
                    "Cannot plan actions without classified intents",
                ],
            }

        planned_actions: list[PlannedAction] = []

        for intent in intents:
            if intent in _TOOL_BY_INTENT:
                planned_actions.append(
                    {
                        "intent": intent,
                        "action_type": ActionType.TOOL,
                        "tool_name": _TOOL_BY_INTENT[intent],
                    }
                )
                continue

            if intent in _RAG_INTENTS:
                planned_actions.append(
                    {
                        "intent": intent,
                        "action_type": ActionType.RAG,
                        "tool_name": None,
                    }
                )
                continue

            # UNKNOWN is intentionally not converted into an action.
            # Existing routing will continue handling it as clarification.
            if intent == Intent.UNKNOWN:
                continue

        print(
            f"[ACTION_PLANNER] "
            f"intents={intents} "
            f"planned_actions={planned_actions}"
        )

        return {
            "planned_actions": planned_actions,
        }

    return plan_actions


def _build_multi_tool_clarification_question(
    *,
    intents: Sequence[Intent],
    missing_fields: Sequence[str],
) -> str:
    """Build a specific clarification question for missing tool arguments."""
    missing = set(missing_fields)

    needs_order_id = any(
        field.endswith(".order_id")
        for field in missing
    )
    needs_category = any(
        field.endswith(".category")
        for field in missing
    )
    needs_description = any(
        field.endswith(".description")
        for field in missing
    )

    requirements: list[str] = []

    if needs_order_id:
        if {
            Intent.ORDER_STATUS,
            Intent.PAYMENT_STATUS,
        }.issubset(intents):
            requirements.append(
                "your order ID to complete the order-status and "
                "payment-status checks"
            )
        else:
            requirements.append("your order ID")

    if needs_category and needs_description:
        requirements.append(
            "the issue category and a description of the problem "
            "to create the support ticket"
        )
    elif needs_category:
        requirements.append(
            "the issue category for the support ticket"
        )
    elif needs_description:
        requirements.append(
            "a description of the issue for the support ticket"
        )

    if not requirements:
        return "Please provide the missing information."

    if len(requirements) == 1:
        requirement_text = requirements[0]
    else:
        requirement_text = "; and ".join(requirements)

    example = " For example: 45821." if needs_order_id else ""

    return f"I need {requirement_text}.{example}"


def _recover_clarification_from_history(
    history: Sequence[ConversationTurn],
) -> str | None:
    """Recover the latest pending clarification without guessing."""
    latest_assistant_message: str | None = None

    for turn in reversed(history):
        if turn["role"] == "assistant":
            latest_assistant_message = turn["content"].strip()
            break

    if not latest_assistant_message:
        return None

    message = latest_assistant_message.lower()

    if message == "please provide the missing information.":
        return None

    clarification_prefixes = (
        "please provide your order id",
        "please provide the issue category",
        "please describe the issue",
        "what category best describes your issue?",
        "i need your order id",
        "i need the issue category",
        "i need a description of the issue",
        "i need the issue category and a description",
    )

    if message.startswith(clarification_prefixes):
        return latest_assistant_message

    return None


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

        history = state.get(
            "conversation_history",
            [],
        )

        try:
            result = classifier.classify(
                message,
                conversation_history=history,
            )
        except IntentClassificationError as exc:
            return {
                "errors": [str(exc)],
                "intent": None,
                "conversation_history": [
                    {
                        "role": "user",
                        "content": message.strip(),
                    }
                ],
            }

        return {
            "intent": result.intent,
            "intents": list(result.intents),
            "conversation_history": [
                {
                    "role": "user",
                    "content": message.strip(),
                }
            ],
        }

    return classify_intent


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
            "conversation_history": [
        {
            "role": "assistant",
            "content": response.answer,
        }
    ],
        }

    return run_rag


def build_clarification_node() -> Callable[[AgentState], dict]:
    """Build a node that returns a context-aware clarification question."""

    def ask_clarification(state: AgentState) -> dict:
        question = state.get("clarification_question")

        if not question:
            question = _recover_clarification_from_history(
                state.get("conversation_history", [])
            )

        if not question and state.get("missing_fields"):
            question = _build_multi_tool_clarification_question(
                intents=state.get("intents", []),
                missing_fields=state.get("missing_fields", []),
            )

        if not question:
            question = "Please provide the missing information."

        print(f"[CLARIFICATION] question={question!r}")

        return {
            "response": question,
            "conversation_history": [
                {
                    "role": "assistant",
                    "content": question,
                }
            ],
        }

    return ask_clarification


def build_tool_selection_node() -> Callable[[AgentState], dict]:
    """Select the business tool from the planner's single TOOL action."""

    def select_tool(state: AgentState) -> dict:
        planned_actions = list(
            state.get("planned_actions") or []
        )

        if len(planned_actions) != 1:
            return {
                "errors": [
                    *state.get("errors", []),
                    (
                        "Single-tool selection requires exactly one "
                        f"planned action, got {len(planned_actions)}"
                    ),
                ],
                "tool_name": None,
            }

        action = planned_actions[0]

        if action["action_type"] != ActionType.TOOL:
            return {
                "errors": [
                    *state.get("errors", []),
                    (
                        "Single-tool selection received a non-tool "
                        f"action: {action['action_type']}"
                    ),
                ],
                "tool_name": None,
            }

        tool_name = action["tool_name"]

        if not tool_name:
            return {
                "errors": [
                    *state.get("errors", []),
                    (
                        "Single-tool planned action does not contain "
                        "a tool name"
                    ),
                ],
                "tool_name": None,
            }

        print(
            "[TOOL_SELECTION] "
            "source=planner "
            f"intent={action['intent']} "
            f"tool_name={tool_name}"
        )

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
            print(
    "[MULTI_TOOL] "
    f"name={tool_name} "
    f"arguments={tool_arguments}"
)
            result = registry.execute(
                tool_name,
                tool_arguments,
            )

            print(
    f"[MULTI_TOOL] name={tool_name} "
    f"arguments={tool_arguments} "
    f"success={result.success} "
    f"error_code={result.error_code} "
    f"error_message={result.error_message} "
    f"data={result.data}"
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
                conversation_history=state.get(
        "conversation_history",
        [],
    ),
            )
            print(
    f"[ARGS] intent={intent} "
    f"message={message!r} "
    f"arguments={result.arguments} "
    f"missing_fields={result.missing_fields} "
    f"clarification={result.clarification_question}"
)
        except ArgumentExtractionError as exc:
            print(f"[ARGS] extraction_failed intent={intent} error={exc}")
            return {
                "errors": [
                    *state.get("errors", []),
                    str(exc),
                ],
                "tool_arguments": {},
                "missing_fields": [],
                "clarification_question": None,
                "route": AgentRoute.ASK_CLARIFICATION,
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


def build_multi_tool_preparation_node(
    extractor: ArgumentExtractor,
) -> Callable[[AgentState], dict]:
    """Prepare and validate arguments from planned tool actions."""

    def prepare_multi_tool_actions(state: AgentState) -> dict:
        planned_actions = list(
            state.get("planned_actions") or []
        )

        if len(planned_actions) < 2:
            return {
                "errors": [
                    *state.get("errors", []),
                    (
                        "Multiple tool preparation requires at least "
                        "two planned actions"
                    ),
                ],
                "planned_tool_actions": [],
            }

        planned_tool_actions: list[dict] = []
        missing_fields: list[str] = []

        for action in planned_actions:
            intent = action["intent"]
            action_type = action["action_type"]
            tool_name = action["tool_name"]

            if action_type != ActionType.TOOL:
                return {
                    "errors": [
                        *state.get("errors", []),
                        (
                            "Multi-tool preparation received a non-tool "
                            f"planned action: {action_type}"
                        ),
                    ],
                    "planned_tool_actions": [],
                }

            if not tool_name:
                return {
                    "errors": [
                        *state.get("errors", []),
                        (
                            "Multi-tool planned action does not contain "
                            "a tool name"
                        ),
                    ],
                    "planned_tool_actions": [],
                }

            if intent == Intent.HUMAN_ESCALATION:
                planned_tool_actions.append(
                    {
                        "intent": intent,
                        "tool_name": tool_name,
                        "tool_arguments": {
                            "reason": state.get(
                                "user_message",
                                "",
                            ).strip(),
                            "priority": "urgent",
                        },
                        "tool_result": None,
                    }
                )
                continue

            try:
                result = extractor.prepare(
                    intent=intent,
                    message=state.get(
                        "user_message",
                        "",
                    ),
                    conversation_history=state.get(
                        "conversation_history",
                        [],
                    ),
                )
            except ArgumentExtractionError as exc:
                return {
                    "errors": [
                        *state.get("errors", []),
                        str(exc),
                    ],
                    "planned_tool_actions": [],
                }

            if result.missing_fields:
                missing_fields.extend(
                    f"{intent.value}.{field}"
                    for field in result.missing_fields
                )

            planned_tool_actions.append(
                {
                    "intent": intent,
                    "tool_name": tool_name,
                    "tool_arguments": result.arguments,
                    "tool_result": None,
                }
            )

        if missing_fields:
            return {
                "planned_tool_actions": planned_tool_actions,
                "missing_fields": missing_fields,
                "clarification_question": (
                    _build_multi_tool_clarification_question(
                        intents=[
                            action["intent"]
                            for action in planned_actions
                        ],
                        missing_fields=missing_fields,
                    )
                ),
            }

        return {
            "planned_tool_actions": planned_tool_actions,
            "missing_fields": [],
            "clarification_question": None,
        }

    return prepare_multi_tool_actions


def build_mixed_action_preparation_node(
    extractor: ArgumentExtractor,
) -> Callable[[AgentState], dict]:
    """Prepare arguments for a mixed RAG and tool workflow."""

    def prepare_mixed_actions(state: AgentState) -> dict:
        actions = list(state.get("planned_actions", []))

        if len(actions) < 2:
            return {
                "errors": [
                    *state.get("errors", []),
                    "Mixed action preparation requires at least two actions",
                ],
                "prepared_actions": [],
            }

        prepared_actions: list[PreparedAction] = []
        missing_fields: list[str] = []

        for action in actions:
            intent = action["intent"]
            action_type = action["action_type"]
            tool_name = action["tool_name"]

            if action_type == ActionType.RAG:
                prepared_actions.append(
                    {
                        "intent": intent,
                        "action_type": ActionType.RAG,
                        "tool_name": None,
                        "tool_arguments": None,
                    }
                )
                continue

            if action_type != ActionType.TOOL or not tool_name:
                return {
                    "errors": [
                        *state.get("errors", []),
                        (
                            "Invalid planned action: "
                            f"intent={intent} action_type={action_type}"
                        ),
                    ],
                    "prepared_actions": [],
                }

            if intent == Intent.HUMAN_ESCALATION:
                prepared_actions.append(
                    {
                        "intent": intent,
                        "action_type": ActionType.TOOL,
                        "tool_name": tool_name,
                        "tool_arguments": {
                            "reason": state.get(
                                "user_message",
                                "",
                            ).strip(),
                            "priority": "urgent",
                        },
                    }
                )
                continue

            try:
                result = extractor.prepare(
                    intent=intent,
                    message=state.get(
                        "user_message",
                        "",
                    ),
                    conversation_history=state.get(
                        "conversation_history",
                        [],
                    ),
                )
            except ArgumentExtractionError as exc:
                return {
                    "errors": [
                        *state.get("errors", []),
                        str(exc),
                    ],
                    "prepared_actions": [],
                }

            if result.missing_fields:
                missing_fields.extend(
                    f"{intent.value}.{field}"
                    for field in result.missing_fields
                )

            prepared_actions.append(
                {
                    "intent": intent,
                    "action_type": ActionType.TOOL,
                    "tool_name": tool_name,
                    "tool_arguments": result.arguments,
                }
            )

        if missing_fields:
            return {
                "prepared_actions": prepared_actions,
                "missing_fields": missing_fields,
                "clarification_question": (
                    _build_multi_tool_clarification_question(
                        intents=[
                            action["intent"]
                            for action in actions
                        ],
                        missing_fields=missing_fields,
                    )
                ),
            }

        return {
            "prepared_actions": prepared_actions,
            "missing_fields": [],
            "clarification_question": None,
        }

    return prepare_mixed_actions


def build_mixed_action_execution_node(
    retriever: Retriever,
    generator: GroundedResponseGenerator,
    registry: ToolRegistry,
) -> Callable[[AgentState], dict]:
    """Execute RAG and business-tool actions in planned order."""

    def execute_mixed_actions(state: AgentState) -> dict:
        actions = list(state.get("prepared_actions", []))

        if not actions:
            return {
                "errors": [
                    *state.get("errors", []),
                    "No prepared actions available",
                ],
                "action_results": [],
            }

        customer_id = state.get("customer_id")
        conversation_id = state.get("conversation_id")
        user_message = state.get("user_message", "")

        action_results: list[ActionExecutionResult] = []
        retrieved_context = []
        sources = []

        for action in actions:
            intent = action["intent"]
            action_type = action["action_type"]
            tool_name = action["tool_name"]

            # -------------------------------------------------
            # RAG action
            # -------------------------------------------------

            if action_type == ActionType.RAG:
                try:
                    results = retriever.retrieve(user_message)

                    retrieved_context.extend(results)

                    rag_response = generator.generate(
                        query=user_message,
                        results=results,
                    )

                    sources.extend(rag_response.sources)

                    action_results.append(
                        {
                            "intent": intent,
                            "action_type": ActionType.RAG,
                            "tool_name": None,
                            "success": True,
                            "response": rag_response.answer,
                            "tool_result": None,
                        }
                    )

                except RetrievalError as exc:
                    print(
                        "[MIXED_ACTION] "
                        f"intent={intent} "
                        f"retrieval_failed={exc}"
                    )

                    action_results.append(
                        {
                            "intent": intent,
                            "action_type": ActionType.RAG,
                            "tool_name": None,
                            "success": False,
                            "response": None,
                            "tool_result": None,
                        }
                    )

                    state_errors = [
                        *state.get("errors", []),
                        str(exc),
                    ]
                    state["errors"] = state_errors

                except GenerationError as exc:
                    print(
                        "[MIXED_ACTION] "
                        f"intent={intent} "
                        f"generation_failed={exc}"
                    )

                    action_results.append(
                        {
                            "intent": intent,
                            "action_type": ActionType.RAG,
                            "tool_name": None,
                            "success": False,
                            "response": None,
                            "tool_result": None,
                        }
                    )

                    state_errors = [
                        *state.get("errors", []),
                        str(exc),
                    ]
                    state["errors"] = state_errors

                continue

            # -------------------------------------------------
            # Tool action
            # -------------------------------------------------

            if action_type != ActionType.TOOL or not tool_name:
                action_results.append(
                    {
                        "intent": intent,
                        "action_type": action_type,
                        "tool_name": tool_name,
                        "success": False,
                        "response": None,
                        "tool_result": None,
                    }
                )

                state_errors = [
                    *state.get("errors", []),
                    (
                        "Invalid mixed action: "
                        f"intent={intent} tool={tool_name}"
                    ),
                ]
                state["errors"] = state_errors
                continue

            tool_arguments = dict(
                action.get("tool_arguments") or {}
            )

            if customer_id is not None:
                tool_arguments["customer_id"] = customer_id

            if tool_name in {
                "create_support_ticket",
                "escalate_to_human",
            } and conversation_id is not None:
                tool_arguments["conversation_id"] = conversation_id

            try:
                print(
                    "[MIXED_ACTION] "
                    f"name={tool_name} "
                    f"arguments={tool_arguments}"
                )

                result = registry.execute(
                    tool_name,
                    tool_arguments,
                )

                print(
                    "[MIXED_ACTION] "
                    f"name={tool_name} "
                    f"success={result.success} "
                    f"error_code={result.error_code} "
                    f"data={result.data}"
                )

            except ToolRegistryError as exc:
                result = ToolResult.failure(
                    error_code="TOOL_EXECUTION_FAILED",
                    error_message=str(exc),
                )

            action_results.append(
                {
                    "intent": intent,
                    "action_type": ActionType.TOOL,
                    "tool_name": tool_name,
                    "success": result.success,
                    "response": None,
                    "tool_result": result.model_dump(),
                }
            )

        return {
            "action_results": action_results,
            "retrieved_context": retrieved_context,
            "sources": sources,
            "errors": state.get("errors", []),
        }

    return execute_mixed_actions

def build_mixed_action_response_node(
    generator: ToolResponseGenerator,
) -> Callable[[AgentState], dict]:
    """Compose one final response from ordered RAG and tool results."""

    def generate_mixed_response(state: AgentState) -> dict:
        action_results = list(
            state.get("action_results", [])
        )

        if not action_results:
            return {
                "errors": [
                    *state.get("errors", []),
                    "No mixed action results available",
                ],
                "response": None,
            }

        user_message = state.get(
            "user_message",
            "",
        )

        responses: list[str] = []

        for action in action_results:
            action_type = action["action_type"]

            # RAG result
            if action_type == ActionType.RAG:
                if action["success"] and action["response"]:
                    responses.append(
                        action["response"].strip()
                    )
                else:
                    responses.append(
                        "I couldn't retrieve the requested "
                        "knowledge-base information right now. "
                        "Please try again."
                    )
                continue

            # Tool result
            if action_type != ActionType.TOOL:
                continue

            tool_name = action["tool_name"]
            tool_result_data = action["tool_result"]

            if not tool_name or not tool_result_data:
                responses.append(
                    "I couldn't complete one part of your request "
                    "right now. Please try again."
                )
                continue

            try:
                tool_result = ToolResult.model_validate(
                    tool_result_data
                )

                tool_response = generator.generate_multiple(
                    user_message=user_message,
                    results=[
                        ToolActionResult(
                            intent=action["intent"],
                            tool_name=tool_name,
                            tool_result=tool_result,
                        )
                    ],
                )

                responses.append(
                    tool_response.answer.strip()
                )

            except ToolResponseGenerationError as exc:
                print(
                    "[MIXED_ACTION] "
                    f"tool_response_failed={exc}"
                )

                responses.append(
                    "I couldn't complete one part of your "
                    "request right now. Please try again."
                )

        response = "\n\n".join(
            item
            for item in responses
            if item
        ).strip()

        if not response:
            return {
                "errors": [
                    *state.get("errors", []),
                    "Unable to generate mixed action response",
                ],
                "response": None,
            }

        return {
            "response": response,
            "conversation_history": [
                {
                    "role": "assistant",
                    "content": response,
                }
            ],
        }

    return generate_mixed_response

def build_multi_tool_execution_node(
    registry: ToolRegistry,
) -> Callable[[AgentState], dict]:
    """Execute multiple validated business tools sequentially."""

    def execute_multi_tool_actions(state: AgentState) -> dict:
        actions = list(
            state.get("planned_tool_actions", [])
        )

        if not actions:
            return {
                "errors": [
                    *state.get("errors", []),
                    "No planned tool actions available",
                ],
                "planned_tool_actions": [],
            }

        customer_id = state.get("customer_id")
        conversation_id = state.get("conversation_id")

        executed_actions = []

        for action in actions:
            tool_name = action["tool_name"]

            tool_arguments = dict(
                action["tool_arguments"]
            )

            if customer_id is not None:
                tool_arguments["customer_id"] = customer_id

            if tool_name in {
                "create_support_ticket",
                "escalate_to_human",
            } and conversation_id is not None:
                tool_arguments["conversation_id"] = conversation_id

            try:
                print(
    "[MULTI_TOOL] "
    f"name={tool_name} "
    f"arguments={tool_arguments}"
)
                result = registry.execute(
                    tool_name,
                    tool_arguments,
                )
                print(
    "[MULTI_TOOL] "
    f"name={tool_name} "
    f"success={result.success} "
    f"error_code={result.error_code} "
    f"data={result.data}"
)
            except ToolRegistryError as exc:
                result = ToolResult.failure(
                    error_code="TOOL_EXECUTION_FAILED",
                    error_message=str(exc),
                )

            executed_actions.append(
                {
                    **action,
                    "tool_arguments": action["tool_arguments"],
                    "tool_result": result.model_dump(),
                }
            )

        return {
            "planned_tool_actions": executed_actions,
        }

    return execute_multi_tool_actions



def build_multi_tool_response_node(
    generator: ToolResponseGenerator,
) -> Callable[[AgentState], dict]:
    """Generate one coherent response from multiple tool results."""

    def generate_multi_tool_response(state: AgentState) -> dict:
        actions = list(
            state.get("planned_tool_actions", [])
        )

        if not actions:
            return {
                "errors": [
                    *state.get("errors", []),
                    "No tool actions available for response generation",
                ],
                "response": None,
            }

        user_message = state.get(
            "user_message",
            "",
        )

        

        tool_results: list[ToolActionResult] = []

        for action in actions:
            try:
                tool_result = ToolResult.model_validate(
                    action["tool_result"]
                )

                tool_results.append(
                    ToolActionResult(
                        intent=action["intent"],
                        tool_name=action["tool_name"],
                        tool_result=tool_result,
                    )
                )

            except ToolResponseGenerationError as exc:
                return {
                    "errors": [
                        *state.get("errors", []),
                        f"Invalid multi-tool result: {exc}",
                    ],
                    "response": None,
                }

        try:
            result = generator.generate_multiple(
                user_message=user_message,
                results=tool_results,
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

    return generate_multi_tool_response


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
            "conversation_history": [
        {
            "role": "assistant",
            "content": result.answer,
        }
    ],
        }

    return generate_tool_response


def build_finalization_node(
    tracer: TurnTracer | None = None,
    guardrails: GuardrailPipeline | None = None,
) -> Callable[[AgentState], dict]:
    """Build the shared end-of-turn finalization node."""

    active_tracer = tracer or NoOpTracing()
    active_guardrails = guardrails or GuardrailPipeline()

    def finalize_turn(state: AgentState) -> dict:
        active_guardrails.check(state)
        active_tracer.on_turn_end(state)

        return {}

    return finalize_turn
