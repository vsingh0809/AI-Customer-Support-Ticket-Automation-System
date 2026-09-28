from decimal import Decimal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.orders import get_order_service, router
from app.core.exceptions import ApplicationServiceError


def build_app(service) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_order_service] = lambda: service
    return app


def test_get_order_returns_200() -> None:
    customer_id = uuid4()
    order_id = uuid4()

    order = type(
        "FakeOrder",
        (),
        {
            "id": order_id,
            "external_order_id": "45821",
            "customer_id": customer_id,
            "status": "shipped",
            "total_amount": Decimal("1499.00"),
            "expected_delivery": None,
        },
    )()

    class FakeOrderService:
        def get_by_external_id(
            self,
            external_order_id,
            requested_customer_id,
        ):
            assert external_order_id == "45821"
            assert requested_customer_id == customer_id
            return order

    client = TestClient(build_app(FakeOrderService()))

    response = client.get(
        "/orders/45821",
        params={"customer_id": str(customer_id)},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["external_order_id"] == "45821"
    assert body["customer_id"] == str(customer_id)
    assert body["status"] == "shipped"
    assert body["total_amount"] == "1499.00"
    assert body["expected_delivery"] is None


def test_get_order_returns_404_when_not_found() -> None:
    class FakeOrderService:
        def get_by_external_id(
            self,
            external_order_id,
            customer_id,
        ):
            return None

    client = TestClient(build_app(FakeOrderService()))

    response = client.get(
        "/orders/45821",
        params={"customer_id": str(uuid4())},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Order not found."


def test_get_order_handles_service_failure() -> None:
    class FailingOrderService:
        def get_by_external_id(
            self,
            external_order_id,
            customer_id,
        ):
            raise ApplicationServiceError("database unavailable")

    client = TestClient(build_app(FailingOrderService()))

    response = client.get(
        "/orders/45821",
        params={"customer_id": str(uuid4())},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to retrieve order."