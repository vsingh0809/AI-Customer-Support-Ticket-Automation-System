from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import Payment
from app.db.repositories import PaymentRepository


class PaymentService:
    def __init__(self, db: Session):
        self.repository = PaymentRepository(db)

    def get_latest_for_order(self, order_id: UUID) -> Payment | None:
        return self.repository.get_latest_for_order(order_id)
