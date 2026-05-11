from __future__ import annotations

from fastapi import APIRouter, Depends

from agents.library.api.router import get_library_repository
from agents.library.repository import LibraryRepository
from agents.main_agent.api.router import get_main_embedding_provider, get_main_llm_client
from agents.main_agent.embedding import EmbeddingProvider
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.repository import MainChunkRepository, get_main_chunk_repository
from agents.orchestrator.api.schemas import (
    OrchestratorChatRequest,
    OrchestratorChatResponse,
    OrchestratorRouteRequest,
    OrchestratorRouteResponse,
)
from agents.orchestrator.routing.evidence import RoutingEvidenceCollector
from agents.orchestrator.service import execute_routed_query, route_query

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])


def get_routing_evidence_collector() -> RoutingEvidenceCollector:
    """FastAPI dependency override가 가능하도록 evidence collector를 분리한다."""
    return RoutingEvidenceCollector()


@router.post("/route", response_model=OrchestratorRouteResponse)
def route(
    request: OrchestratorRouteRequest,
    evidence_collector: RoutingEvidenceCollector = Depends(get_routing_evidence_collector),
) -> OrchestratorRouteResponse:
    """Spring Core가 호출하는 evidence 기반 route endpoint."""
    return route_query(request, evidence_collector=evidence_collector)


@router.post("/chat", response_model=None)
def chat(
    request: OrchestratorChatRequest,
    evidence_collector: RoutingEvidenceCollector = Depends(get_routing_evidence_collector),
    main_repository: MainChunkRepository = Depends(get_main_chunk_repository),
    main_embedding_provider: EmbeddingProvider = Depends(get_main_embedding_provider),
    main_llm_client: LLMClient = Depends(get_main_llm_client),
    library_repository: LibraryRepository = Depends(get_library_repository),
) -> OrchestratorChatResponse:
    """질의를 라우팅한 뒤 선택된 내부 Agent를 실행해 최종 응답을 반환한다."""
    return execute_routed_query(
        request,
        evidence_collector=evidence_collector,
        main_repository=main_repository,
        main_embedding_provider=main_embedding_provider,
        main_llm_client=main_llm_client,
        library_repository=library_repository,
    )
