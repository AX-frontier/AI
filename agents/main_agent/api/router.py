from __future__ import annotations

from fastapi import APIRouter, Depends

from agents.main_agent.agent import run_main_agent
from agents.main_agent.api.schemas import MainChatRequest, MainChatResponse
from agents.main_agent.embedding import EmbeddingProvider, get_embedding_provider
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.llm.factory import get_llm_client
from agents.main_agent.repository import MainChunkRepository, get_main_chunk_repository

router = APIRouter(prefix="/main", tags=["main-agent"])


def get_main_embedding_provider() -> EmbeddingProvider:
    return get_embedding_provider()


def get_main_llm_client() -> LLMClient:
    return get_llm_client()


@router.post("/chat", response_model=MainChatResponse)
def chat(
    request: MainChatRequest,
    repository: MainChunkRepository = Depends(get_main_chunk_repository),
    embedding_provider: EmbeddingProvider = Depends(get_main_embedding_provider),
    llm_client: LLMClient = Depends(get_main_llm_client),
) -> MainChatResponse:
    """Spring에서 호출하는 Main Agent 학교 정보 QA endpoint."""
    return run_main_agent(
        request,
        repository=repository,
        embedding_provider=embedding_provider,
        llm_client=llm_client,
    )
