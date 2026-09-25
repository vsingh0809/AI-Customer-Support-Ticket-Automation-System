from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Order


class OrderRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_external_id(
        self, external_order_id: str, customer_id: UUID | None = None
    ) -> Order | None:
        statement = select(Order).where(Order.external_order_id == external_order_id)
        if customer_id is not None:
            statement = statement.where(Order.customer_id == customer_id)
        return self.db.scalar(statement)
