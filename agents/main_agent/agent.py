from __future__ import annotations

from agents.main_agent.api.schemas import MainChatRequest, MainChatResponse
from agents.main_agent.embedding import EmbeddingProvider, get_embedding_provider
from agents.main_agent.generator import build_main_response
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.llm.factory import get_llm_client
from agents.main_agent.page_navigation import build_page_navigation_response
from agents.main_agent.repository import MainChunkRepository, get_main_chunk_repository
from agents.main_agent.retrieval import MainRetriever


def run_main_agent(
    request: MainChatRequest,
    repository: MainChunkRepository | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    llm_client: LLMClient | None = None,
) -> MainChatResponse:
    """Spring 요청을 받아 Main Agent RAG 검색/응답 파이프라인을 실행한다."""
    page_navigation_response = build_page_navigation_response(request.message)
    if page_navigation_response is not None:
        return page_navigation_response

    repo = repository or get_main_chunk_repository()
    embedder = embedding_provider or get_embedding_provider()
    llm = llm_client or get_llm_client()
    retriever = MainRetriever(repo, embedder)

    result = retriever.retrieve(request.message.strip())
    return build_main_response(result, llm_client=llm)
