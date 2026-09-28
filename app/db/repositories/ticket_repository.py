from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Ticket


class TicketRepository:
    def __init__(self, db: Session):
        self.db = db

    def add(self, ticket: Ticket) -> Ticket:
        self.db.add(ticket)
        self.db.flush()
        return ticket

    def get_by_id(
        self,
        *,
        ticket_id: UUID,
        customer_id: UUID,
    ) -> Ticket | None:
        statement = select(Ticket).where(
            Ticket.id == ticket_id,
            Ticket.customer_id == customer_id,
        )
        return self.db.scalar(statement)