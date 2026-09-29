from decimal import Decimal
from uuid import UUID

from sqlalchemy import select

from app.db.models import Customer, Order, OrderItem, Payment
from app.db.session import SessionLocal

DEMO_CUSTOMER_ID = UUID(
    "3f2b8f18-7f3d-4d6e-9a2c-5c3e4a7d8b1f"
)
DEMO_ORDER_ID = UUID(
    "6b5c3d2e-1f90-4a72-8b44-2e8f7c1a9d35"
)
DEMO_ORDER_ITEM_ID = UUID(
    "7c6d4e3f-2a81-4b63-9c55-3f9e8d2b1a46"
)
DEMO_PAYMENT_ID = UUID(
    "8a71c6d4-42e9-4b13-91f5-7c2d8e6a4031"
)

DEMO_EMAIL = "rahul@example.com"
DEMO_NAME = "Rahul Singh"
DEMO_EXTERNAL_ORDER_ID = "45821"
DEMO_TRANSACTION_ID = "TXN-DEMO-45821"


def seed() -> None:
    with SessionLocal() as db:
        customer = db.scalar(
            select(Customer).where(
                Customer.email == DEMO_EMAIL
            )
        )

        if customer is not None:
            if customer.id != DEMO_CUSTOMER_ID:
                raise RuntimeError(
                    "Legacy demo seed detected. "
                    "Reset the development database before running "
                    "the canonical seed."
                )

            customer.full_name = DEMO_NAME
        else:
            customer = Customer(
                id=DEMO_CUSTOMER_ID,
                email=DEMO_EMAIL,
                full_name=DEMO_NAME,
            )
            db.add(customer)
            db.flush()

        order = db.scalar(
            select(Order).where(
                Order.external_order_id
                == DEMO_EXTERNAL_ORDER_ID
            )
        )

        if order is None:
            order = Order(
                id=DEMO_ORDER_ID,
                customer_id=DEMO_CUSTOMER_ID,
                external_order_id=DEMO_EXTERNAL_ORDER_ID,
                status="shipped",
                total_amount=Decimal("1499.00"),
            )
            db.add(order)
            db.flush()
        else:
            if order.customer_id != DEMO_CUSTOMER_ID:
                raise RuntimeError(
                    "Canonical demo order belongs to a different "
                    "customer. Reset the development database."
                )

            order.status = "shipped"
            order.total_amount = Decimal("1499.00")
            order.expected_delivery = None

        order_item = db.scalar(
            select(OrderItem).where(
                OrderItem.order_id == order.id
            )
        )

        if order_item is None:
            db.add(
                OrderItem(
                    id=DEMO_ORDER_ITEM_ID,
                    order_id=order.id,
                    product_name="Wireless Headphones",
                    quantity=1,
                    unit_price=Decimal("1499.00"),
                )
            )
        else:
            order_item.product_name = "Wireless Headphones"
            order_item.quantity = 1
            order_item.unit_price = Decimal("1499.00")

        payment = db.scalar(
            select(Payment).where(
                Payment.order_id == order.id
            )
        )

        if payment is None:
            db.add(
                Payment(
                    id=DEMO_PAYMENT_ID,
                    order_id=order.id,
                    transaction_id=DEMO_TRANSACTION_ID,
                    status="paid",
                    amount=Decimal("1499.00"),
                )
            )
        else:
            payment.transaction_id = DEMO_TRANSACTION_ID
            payment.status = "paid"
            payment.amount = Decimal("1499.00")

        db.commit()

        print("Canonical demo seed created/updated.")
        print(f"Customer ID: {DEMO_CUSTOMER_ID}")
        print(f"Order ID: {DEMO_EXTERNAL_ORDER_ID}")


if __name__ == "__main__":
    seed()