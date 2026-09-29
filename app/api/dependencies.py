"""FastAPI dependency providers and application composition."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.ai.agent.argument_extractor import ArgumentExtractor
from app.ai.agent.graph import build_agent_graph
from app.ai.agent.intent_classifier import IntentClassifier
from app.ai.agent.router import AgentRouter
from app.ai.agent.tool_response import ToolResponseGenerator
from app.ai.rag.embeddings import EmbeddingService, FastEmbedProvider
from app.ai.rag.generator import DeepSeekProvider, GroundedResponseGenerator
from app.ai.rag.retriever import Retriever
from app.ai.rag.vector_store import QdrantVectorStore
from app.ai.tools.escalation_tools import EscalationTools
from app.ai.tools.order_tools import OrderTools
from app.ai.tools.payment_tools import PaymentTools
from app.ai.tools.registry import ToolRegistry
from app.ai.tools.ticket_tools import TicketTools
from app.core.config import get_settings
from app.db.repositories import (
    ConversationRepository,
    MessageRepository,
)
from app.db.session import get_db
from app.services.chat_application import ChatApplicationService
from app.services.conversation_memory import ConversationMemoryService
from app.services.order_service import OrderService
from app.services.payment_service import PaymentService
from app.services.ticket_service import TicketService


class _ToolResponseProviderAdapter:
    """Adapt DeepSeekProvider to ToolResponseGenerator's provider contract."""

    def __init__(self, provider: DeepSeekProvider) -> None:
        self._provider = provider

    def generate(self, prompt: str) -> str:
        return self._provider.generate(
            system_prompt=(
                "Generate a concise customer-support response "
                "using only the tool result provided in the user prompt."
            ),
            user_prompt=prompt,
        )


@dataclass(frozen=True)
class StaticAgentDependencies:
    """AI dependencies that can safely be shared across requests."""

    classifier: IntentClassifier
    router: AgentRouter
    retriever: Retriever
    generator: GroundedResponseGenerator
    argument_extractor: ArgumentExtractor
    tool_response_generator: ToolResponseGenerator


@lru_cache
def get_static_agent_dependencies() -> StaticAgentDependencies:
    """Build expensive AI dependencies once for the application."""

    settings = get_settings()

    if not settings.deepseek_api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is required to start the AI agent."
        )

    if not settings.qdrant_url:
        raise RuntimeError(
            "QDRANT_URL is required to start the RAG system."
        )

    llm_provider = DeepSeekProvider(
        api_key=settings.deepseek_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    embedding_provider = FastEmbedProvider(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    embedding_service = EmbeddingService(
        embedding_provider,
        batch_size=settings.embedding_batch_size,
    )

    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )

    vector_store = QdrantVectorStore(
        qdrant_client,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.qdrant_vector_size,
    )

    vector_store.ensure_collection()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    return StaticAgentDependencies(
        classifier=IntentClassifier(llm_provider),
        router=AgentRouter(),
        retriever=retriever,
        generator=GroundedResponseGenerator(llm_provider),
        argument_extractor=ArgumentExtractor(llm_provider),
        tool_response_generator=ToolResponseGenerator(
            _ToolResponseProviderAdapter(llm_provider)
        ),
    )


def get_conversation_memory_service(
    db: Annotated[Session, Depends(get_db)],
) -> ConversationMemoryService:
    """Build the conversation-memory service for the current request."""

    return ConversationMemoryService(
        conversation_repository=ConversationRepository(db),
        message_repository=MessageRepository(db),
    )


def get_chat_application_service(
    db: Annotated[Session, Depends(get_db)],
) -> ChatApplicationService:
    """Build the chat application with request-scoped business tools."""

    static_dependencies = get_static_agent_dependencies()

    order_service = OrderService(db)
    payment_service = PaymentService(db)
    ticket_service = TicketService(db)

    tool_registry = ToolRegistry()

    tool_registry.register(
        OrderTools(order_service)
    )
    tool_registry.register(
        PaymentTools(
            order_service=order_service,
            payment_service=payment_service,
        )
    )
    tool_registry.register(
        TicketTools(ticket_service)
    )
    tool_registry.register(
        EscalationTools(ticket_service)
    )

    agent_graph = build_agent_graph(
        classifier=static_dependencies.classifier,
        router=static_dependencies.router,
        retriever=static_dependencies.retriever,
        generator=static_dependencies.generator,
        tool_registry=tool_registry,
        argument_extractor=static_dependencies.argument_extractor,
        tool_response_generator=static_dependencies.tool_response_generator,
    )

    memory_service = get_conversation_memory_service(db)

    return ChatApplicationService(
        memory_service=memory_service,
        agent_graph=agent_graph,
    )