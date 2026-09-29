from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.api.routes.conversations import (
    router as conversations_router,
)
from app.api.routes.escalation import router as escalation_router
from app.api.routes.health import router as health_router
from app.api.routes.orders import router as orders_router
from app.api.routes.payments import router as payments_router
from app.api.routes.tickets import router as tickets_router
from app.core.config import get_settings


settings = get_settings()

app = FastAPI(title=settings.app_name)

app.include_router(health_router)

app.include_router(chat_router)

app.include_router(conversations_router)

app.include_router(tickets_router)

app.include_router(orders_router)

app.include_router(payments_router)

app.include_router(escalation_router)