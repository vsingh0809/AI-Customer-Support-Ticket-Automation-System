from app.ai.agent.argument_extractor import ArgumentExtractor
from app.ai.agent.contracts import AgentRoute, Intent
from app.ai.agent.nodes import build_argument_preparation_node


class FakeProvider:
    def __init__(self, response):
        self.response = response

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ):
        return self.response


def test_argument_node_prepares_order_arguments() -> None:
    extractor = ArgumentExtractor(
        FakeProvider(
            {
                "order_id": "45821",
                "category": None,
                "description": None,
                "reason": None,
                "priority": None,
            }
        )
    )

    node = build_argument_preparation_node(extractor)

    result = node(
        {
            "intent": Intent.ORDER_STATUS,
            "route": AgentRoute.TOOL,
            "user_message": "Where is my order 45821?",
            "errors": [],
        }
    )

    assert result["tool_arguments"] == {
        "order_id": "45821",
    }

    assert result["missing_fields"] == []
    assert result["route"] == AgentRoute.TOOL


def test_argument_node_changes_route_when_information_is_missing() -> None:
    extractor = ArgumentExtractor(
        FakeProvider(
            {
                "order_id": None,
                "category": None,
                "description": None,
                "reason": None,
                "priority": None,
            }
        )
    )

    node = build_argument_preparation_node(extractor)

    result = node(
        {
            "intent": Intent.ORDER_STATUS,
            "route": AgentRoute.TOOL,
            "user_message": "Where is my order?",
            "errors": [],
        }
    )

    assert result["tool_arguments"] == {}
    assert result["missing_fields"] == ["order_id"]
    assert result["route"] == AgentRoute.ASK_CLARIFICATION
    assert result["clarification_question"] == (
        "Please provide your order ID so I can check it."
    )