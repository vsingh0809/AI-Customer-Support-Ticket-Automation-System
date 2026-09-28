from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.tickets import get_ticket_service, router
from app.core.exceptions import ApplicationServiceError


def build_app(service) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_ticket_service] = lambda: service
    return app


def test_create_ticket_returns_201() -> None:
    customer_id = uuid4()
    service = type("FakeTicketService", (), {})()

    def create(**kwargs):
        return type(
            "FakeTicket",
            (),
            {
                "id": uuid4(),
                "customer_id": customer_id,
                "category": kwargs["category"],
                "description": kwargs["description"],
                "priority": kwargs["priority"],
                "status": "open",
                "conversation_id": kwargs["conversation_id"],
            },
        )()

    service.create = create

    client = TestClient(build_app(service))

    response = client.post(
        "/tickets",
        json={
            "customer_id": str(customer_id),
            "category": "refund",
            "description": "I received a damaged product.",
            "priority": "high",
        },
    )

    assert response.status_code == 201

    body = response.json()

    assert body["customer_id"] == str(customer_id)
    assert body["category"] == "refund"
    assert body["priority"] == "high"
    assert body["status"] == "open"


def test_create_ticket_rejects_invalid_priority() -> None:
    client = TestClient(build_app(object()))

    response = client.post(
        "/tickets",
        json={
            "customer_id": str(uuid4()),
            "category": "refund",
            "description": "Refund required.",
            "priority": "critical",
        },
    )

    assert response.status_code == 422


def test_create_ticket_handles_service_failure() -> None:
    class FailingService:
        def create(self, **kwargs):
            raise ApplicationServiceError("database failure")

    client = TestClient(build_app(FailingService()))

    response = client.post(
        "/tickets",
        json={
            "customer_id": str(uuid4()),
            "category": "refund",
            "description": "Refund required.",
            "priority": "normal",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to create support ticket."

def test_get_ticket_returns_200() -> None:
    customer_id = uuid4()
    ticket_id = uuid4()

    class FakeTicketService:
        def get_by_id(self, **kwargs):
            return type(
                "FakeTicket",
                (),
                {
                    "id": ticket_id,
                    "customer_id": customer_id,
                    "category": "refund",
                    "description": "Damaged product.",
                    "priority": "high",
                    "status": "open",
                    "conversation_id": None,
                },
            )()

    client = TestClient(build_app(FakeTicketService()))

    response = client.get(
        f"/tickets/{ticket_id}",
        params={"customer_id": str(customer_id)},
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(ticket_id)


def test_get_ticket_returns() -> None:
    customer_id = uuid4()
    ticket_id = uuid4()

    class FakeTicketService:
        def get_by_id(self, **kwargs):
            return type(
                "FakeTicket",
                (),
                {
                    "id": ticket_id,
                    "customer_id": customer_id,
                    "category": "refund",
                    "description": "Damaged product.",
                    "priority": "high",
                    "status": "open",
                    "conversation_id": None,
                },
            )()

    client = TestClient(build_app(FakeTicketService()))

    response = client.get(
        f"/tickets/{ticket_id}",
        params={"customer_id": str(customer_id)},
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(ticket_id)


def test_get_ticket_returns_404_for_unknown_ticket() -> None:
    class FakeTicketService:
        def get_by_id(self, **kwargs):
            return None

    client = TestClient(build_app(FakeTicketService()))

    response = client.get(
        f"/tickets/{uuid4()}",
        params={"customer_id": str(uuid4())},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Support ticket not found."
