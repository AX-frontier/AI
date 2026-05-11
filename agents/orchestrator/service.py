from __future__ import annotations

from agents.document_review.agent import run_document_review_agent
from agents.document_review.api.schemas import DocumentReviewRequest
from agents.library.agent import run_library_agent
from agents.library.api.schemas import LibraryChatRequest
from agents.library.repository import LibraryRepository
from agents.main_agent.agent import run_main_agent
from agents.main_agent.api.schemas import MainChatRequest
from agents.main_agent.embedding import EmbeddingProvider
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.repository import MainChunkRepository
from agents.orchestrator.api.schemas import (
    OrchestratorChatRequest,
    OrchestratorChatResponse,
    OrchestratorFallbackResponse,
    OrchestratorRouteRequest,
    OrchestratorRouteResponse,
    RoutingEvidencePayload,
)
from agents.orchestrator.routing.evidence import RoutingEvidenceCollector
from agents.orchestrator.routing.router import EvidenceBasedRouter


def route_query(
    request: OrchestratorRouteRequest,
    *,
    evidence_collector: RoutingEvidenceCollector | None = None,
    router: EvidenceBasedRouter | None = None,
) -> OrchestratorRouteResponse:
    """질의를 evidence 기반으로 라우팅하고 Spring 호환 응답을 반환한다."""
    collector = evidence_collector or RoutingEvidenceCollector()
    route_router = router or EvidenceBasedRouter()
    evidence = collector.collect(request.message)
    decision = route_router.route(evidence)

    return OrchestratorRouteResponse(
        queryUid=request.queryUid,
        traceId=request.traceId,
        conversationUid=request.conversationUid,
        targetAgent=decision.target_agent,
        intent=decision.intent,
        confidence=round(decision.confidence, 3),
        reason=decision.reason,
        evidence=RoutingEvidencePayload(
            mainScore=evidence.main.score,
            libraryScore=evidence.library.score,
            documentReviewScore=evidence.document_review.score,
            mainReason=evidence.main.reason,
            libraryReason=evidence.library.reason,
            documentReviewReason=evidence.document_review.reason,
        ),
    )


def execute_routed_query(
    request: OrchestratorChatRequest,
    *,
    evidence_collector: RoutingEvidenceCollector | None = None,
    router: EvidenceBasedRouter | None = None,
    main_repository: MainChunkRepository | None = None,
    main_embedding_provider: EmbeddingProvider | None = None,
    main_llm_client: LLMClient | None = None,
    library_repository: LibraryRepository | None = None,
) -> OrchestratorChatResponse:
    """질의를 라우팅한 뒤 선택된 Agent를 내부 함수 호출로 바로 실행한다."""
    route_result = route_query(
        OrchestratorRouteRequest(
            queryUid=request.queryUid,
            traceId=request.traceId,
            conversationUid=request.conversationUid,
            message=request.message,
        ),
        evidence_collector=evidence_collector,
        router=router,
    )

    if route_result.targetAgent == "MAIN":
        return run_main_agent(
            MainChatRequest(
                queryUid=request.queryUid,
                traceId=request.traceId,
                conversationUid=request.conversationUid,
                message=request.message,
            ),
            repository=main_repository,
            embedding_provider=main_embedding_provider,
            llm_client=main_llm_client,
        )

    if route_result.targetAgent == "LIBRARY":
        return run_library_agent(
            LibraryChatRequest.model_construct(
                queryUid=request.queryUid,
                traceId=request.traceId,
                conversationUid=request.conversationUid,
                message=request.message,
            ),
            repository=library_repository,
        )

    if route_result.targetAgent == "DOCUMENT_REVIEW":
        if request.document is None or not request.document.bodyText.strip():
            return _build_fallback_response(
                "문서 검토 대상 본문이 없어 document-review를 실행할 수 없습니다."
            )
        return run_document_review_agent(
            DocumentReviewRequest(
                queryUid=request.queryUid,
                traceId=request.traceId,
                conversationUid=request.conversationUid,
                message=request.message,
                document=request.document,
            )
        )

    return _build_fallback_response("질문을 이해하지 못했습니다. 관련 주제로 다시 입력해 주세요.")


def _build_fallback_response(reason: str) -> OrchestratorFallbackResponse:
    return OrchestratorFallbackResponse(
        answer=reason,
        fallbackReason=reason,
    )
