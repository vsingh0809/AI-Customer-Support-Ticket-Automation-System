from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Message


class MessageRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_by_conversation(
        self,
        conversation_id: UUID,
    ) -> list[Message]:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )

        return list(self.db.scalars(statement).all())

    def add(
        self,
        message: Message,
    ) -> Message:
        self.db.add(message)
        self.db.flush()
        return message