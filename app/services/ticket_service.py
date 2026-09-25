from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.exceptions import ApplicationServiceError
from app.db.models import Ticket
from app.db.repositories import TicketRepository


class TicketService:
    def __init__(self, db: Session):
        self.repository = TicketRepository(db)
        self.db = db

    def create(
        self,
        *,
        customer_id: UUID,
        category: str,
        description: str,
        priority: str = "normal",
        conversation_id: UUID | None = None,
    ) -> Ticket:
        ticket = Ticket(
            customer_id=customer_id,
            conversation_id=conversation_id,
            category=category,
            description=description,
            priority=priority,
        )

        try:
            self.repository.add(ticket)
            self.db.commit()
            self.db.refresh(ticket)
            return ticket

        except SQLAlchemyError as exc:
            self.db.rollback()

            raise ApplicationServiceError(
                "Failed to create support ticket"
            ) from exc