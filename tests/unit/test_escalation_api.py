from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ai.tools.base import ToolResult
from app.api.routes.escalation import get_escalation_tool, router


def build_app(tool) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_escalation_tool] = lambda: tool
    return app


def test_escalate_returns_201() -> None:
    customer_id = uuid4()
    ticket_id = uuid4()
    conversation_id = uuid4()

    class FakeEscalationTool:
        def escalate_to_human(self, **kwargs):
            assert kwargs["customer_id"] == customer_id
            assert kwargs["reason"] == "I need a human agent."
            assert kwargs["priority"] == "urgent"
            assert kwargs["conversation_id"] == conversation_id

            return ToolResult.ok(
                {
                    "escalated": True,
                    "ticket_id": str(ticket_id),
                    "category": "human_escalation",
                    "priority": "urgent",
                    "status": "open",
                    "reason": "I need a human agent.",
                    "conversation_id": str(conversation_id),
                }
            )

    client = TestClient(build_app(FakeEscalationTool()))

    response = client.post(
        "/escalate",
        json={
            "customer_id": str(customer_id),
            "reason": "I need a human agent.",
            "priority": "urgent",
            "conversation_id": str(conversation_id),
        },
    )

    assert response.status_code == 201

    body = response.json()

    assert body["id"] == str(ticket_id)
    assert body["customer_id"] == str(customer_id)
    assert body["category"] == "human_escalation"
    assert body["priority"] == "urgent"
    assert body["status"] == "open"


def test_escalate_rejects_invalid_priority() -> None:
    client = TestClient(build_app(object()))

    response = client.post(
        "/escalate",
        json={
            "customer_id": str(uuid4()),
            "reason": "I need a human.",
            "priority": "low",
        },
    )

    assert response.status_code == 422


def test_escalate_handles_tool_failure() -> None:
    class FailingEscalationTool:
        def escalate_to_human(self, **kwargs):
            return ToolResult.failure(
                error_code="ESCALATION_FAILED",
                error_message="Unable to escalate the issue.",
            )

    client = TestClient(build_app(FailingEscalationTool()))

    response = client.post(
        "/escalate",
        json={
            "customer_id": str(uuid4()),
            "reason": "I need a human.",
            "priority": "urgent",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to escalate the issue."