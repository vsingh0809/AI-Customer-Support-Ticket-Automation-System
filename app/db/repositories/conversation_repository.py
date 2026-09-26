from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Conversation


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(
        self,
        conversation_id: UUID,
        customer_id: UUID | None = None,
    ) -> Conversation | None:
        statement = select(Conversation).where(
            Conversation.id == conversation_id
        )

        if customer_id is not None:
            statement = statement.where(
                Conversation.customer_id == customer_id
            )

        return self.db.scalar(statement)

    def add(
        self,
        conversation: Conversation,
    ) -> Conversation:
        self.db.add(conversation)
        self.db.flush()
        return conversation