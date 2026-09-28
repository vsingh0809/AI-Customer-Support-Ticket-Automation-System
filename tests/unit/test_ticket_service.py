from unittest.mock import Mock
from uuid import uuid4

from app.services.ticket_service import TicketService


def test_get_by_id_returns_customer_ticket() -> None:
    db = Mock()
    repository = Mock()

    ticket_id = uuid4()
    customer_id = uuid4()

    ticket = Mock()
    ticket.id = ticket_id
    ticket.customer_id = customer_id

    repository.get_by_id.return_value = ticket

    service = TicketService(db)
    service.repository = repository

    result = service.get_by_id(
        ticket_id=ticket_id,
        customer_id=customer_id,
    )

    assert result is ticket

    repository.get_by_id.assert_called_once_with(
        ticket_id=ticket_id,
        customer_id=customer_id,
    )


def test_get_by_id_returns_none_when_ticket_not_found() -> None:
    db = Mock()
    repository = Mock()

    repository.get_by_id.return_value = None

    service = TicketService(db)
    service.repository = repository

    result = service.get_by_id(
        ticket_id=uuid4(),
        customer_id=uuid4(),
    )

    assert result is None    