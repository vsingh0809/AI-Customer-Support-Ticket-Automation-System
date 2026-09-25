from uuid import UUID

from sqlalchemy.orm import Session

from app.core.exceptions import ApplicationServiceError
from app.db.models import Payment
from app.db.repositories import PaymentRepository


class PaymentService:
    def __init__(self, db: Session):
        self.repository = PaymentRepository(db)

    def get_latest_for_order(
        self,
        order_id: UUID,
    ) -> Payment | None:
        try:
            return self.repository.get_latest_for_order(order_id)
        except Exception as exc:
            raise ApplicationServiceError(
                "Failed to retrieve payment"
            ) from exc