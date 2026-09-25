from decimal import Decimal

from sqlalchemy import select

from app.db.models import Customer, Order, OrderItem, Payment
from app.db.session import SessionLocal


def seed() -> None:
    with SessionLocal() as db:
        existing = db.scalar(select(Customer).where(Customer.email == "demo@example.com"))
        if existing:
            print("Seed data already exists")
            return

        customer = Customer(email="demo@example.com", full_name="Demo Customer")
        db.add(customer)
        db.flush()

        order = Order(
            customer_id=customer.id,
            external_order_id="45821",
            status="shipped",
            total_amount=Decimal("1499.00"),
        )
        db.add(order)
        db.flush()

        db.add(
            OrderItem(
                order_id=order.id,
                product_name="Wireless Headphones",
                quantity=1,
                unit_price=Decimal("1499.00"),
            )
        )
        db.add(
            Payment(
                order_id=order.id,
                transaction_id="TXN-DEMO-45821",
                status="paid",
                amount=Decimal("1499.00"),
            )
        )
        db.commit()
        print("Seed data created")


if __name__ == "__main__":
    seed()
