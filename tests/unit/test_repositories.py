from uuid import uuid4
from unittest.mock import Mock

from app.db.models import Order
from app.db.repositories import OrderRepository


def test_order_repository_filters_by_customer() -> None:
    db = Mock()
    scalar = Mock(return_value=None)
    db.scalar = scalar

    repository = OrderRepository(db)
    customer_id = uuid4()
    repository.get_by_external_id("45821", customer_id)

    statement = scalar.call_args.args[0]
    compiled = str(statement.compile(compile_kwargs={"literal_binds": False}))

    assert 'orders.external_order_id' in compiled
    assert 'orders.customer_id' in compiled


def test_order_model_uses_external_order_id() -> None:
    order = Order(external_order_id="45821", status="shipped", total_amount=1499)
    assert order.external_order_id == "45821"
    assert order.status == "shipped"
