from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Payment


class PaymentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_latest_for_order(self, order_id: UUID) -> Payment | None:
        statement = (
            select(Payment)
            .where(Payment.order_id == order_id)
            .order_by(Payment.created_at.desc())
            .limit(1)
        )
        return self.db.scalar(statement)
