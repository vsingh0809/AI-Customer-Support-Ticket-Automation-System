from decimal import Decimal
from uuid import uuid4
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.payments import (
    get_order_service,
    get_payment_service,
    router,
)
from app.core.exceptions import ApplicationServiceError


def build_app(order_service, payment_service) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_order_service] = (
        lambda: order_service
    )
    app.dependency_overrides[get_payment_service] = (
        lambda: payment_service
    )

    return app


def test_get_payment_returns_200() -> None:
    customer_id = uuid4()
    order_db_id = uuid4()
    payment_id = uuid4()

    order = type(
        "FakeOrder",
        (),
        {
            "id": order_db_id,
            "external_order_id": "45821",
            "customer_id": customer_id,
        },
    )()

    payment = type(
        "FakePayment",
        (),
        {
            "id": payment_id,
            "order_id": order_db_id,
            "transaction_id": "txn-45821",
            "status": "successful",
            "amount": Decimal("1499.00"),
            "created_at": None,
        },
    )()

    class FakeOrderService:
        def get_by_external_id(
            self,
            external_order_id,
            requested_customer_id:uuid,
        ):
            assert external_order_id == "45821"
            assert requested_customer_id == customer_id
            return order

    class FakePaymentService:
        def get_latest_for_order(self, order_id):
            assert order_id == order_db_id
            return payment

    client = TestClient(
        build_app(
            FakeOrderService(),
            FakePaymentService(),
        )
    )

    response = client.get(
        "/payments/45821",
        params={"customer_id": str(customer_id)},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["payment_id"] == str(payment_id)
    assert body["order_id"] == "45821"
    assert body["transaction_id"] == "txn-45821"
    assert body["status"] == "successful"
    assert body["amount"] == "1499.00"
    assert body["created_at"] is None


def test_get_payment_returns_404_when_order_not_found() -> None:
    class FakeOrderService:
        def get_by_external_id(
            self,
            external_order_id,
            customer_id,
        ):
            return None

    class FakePaymentService:
        def get_latest_for_order(self, order_id):
            raise AssertionError("Should not be called")

    client = TestClient(
        build_app(
            FakeOrderService(),
            FakePaymentService(),
        )
    )

    response = client.get(
        "/payments/45821",
        params={"customer_id": str(uuid4())},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Order not found."


def test_get_payment_returns_404_when_payment_not_found() -> None:
    order = type(
        "FakeOrder",
        (),
        {
            "id": uuid4(),
            "external_order_id": "45821",
        },
    )()

    class FakeOrderService:
        def get_by_external_id(
            self,
            external_order_id,
            customer_id,
        ):
            return order

    class FakePaymentService:
        def get_latest_for_order(self, order_id):
            return None

    client = TestClient(
        build_app(
            FakeOrderService(),
            FakePaymentService(),
        )
    )

    response = client.get(
        "/payments/45821",
        params={"customer_id": str(uuid4())},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Payment not found."


def test_get_payment_handles_payment_service_failure() -> None:
    order = type(
        "FakeOrder",
        (),
        {
            "id": uuid4(),
            "external_order_id": "45821",
        },
    )()

    class FakeOrderService:
        def get_by_external_id(
            self,
            external_order_id,
            customer_id,
        ):
            return order

    class FailingPaymentService:
        def get_latest_for_order(self, order_id):
            raise ApplicationServiceError("database unavailable")

    client = TestClient(
        build_app(
            FakeOrderService(),
            FailingPaymentService(),
        )
    )

    response = client.get(
        "/payments/45821",
        params={"customer_id": str(uuid4())},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to retrieve payment."