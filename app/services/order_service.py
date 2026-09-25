from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import Order
from app.db.repositories import OrderRepository


class OrderService:
    def __init__(self, db: Session):
        self.repository = OrderRepository(db)

    def get_by_external_id(self, external_order_id: str, customer_id: UUID | None = None) -> Order | None:
        return self.repository.get_by_external_id(external_order_id, customer_id)
