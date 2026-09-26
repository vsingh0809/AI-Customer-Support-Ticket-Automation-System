from fastapi import APIRouter

from app.api.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    raise NotImplementedError("Chat execution will be implemented in Phase 6.2")