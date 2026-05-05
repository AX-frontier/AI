from __future__ import annotations

from fastapi import APIRouter, Depends

from agents.main_agent.agent import run_main_agent
from agents.main_agent.api.schemas import MainChatRequest, MainChatResponse
from agents.main_agent.repository import MainChunkRepository, get_main_chunk_repository

router = APIRouter(prefix="/main", tags=["main-agent"])


@router.post("/chat", response_model=MainChatResponse)
def chat(
    request: MainChatRequest,
    repository: MainChunkRepository = Depends(get_main_chunk_repository),
) -> MainChatResponse:
    """Spring에서 호출하는 Main Agent 학교 정보 QA endpoint."""
    return run_main_agent(request, repository=repository)
