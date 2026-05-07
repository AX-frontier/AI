from __future__ import annotations

from agents.orchestrator.api.schemas import (
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
