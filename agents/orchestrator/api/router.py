from __future__ import annotations

from fastapi import APIRouter, Depends

from agents.orchestrator.api.schemas import OrchestratorRouteRequest, OrchestratorRouteResponse
from agents.orchestrator.routing.evidence import RoutingEvidenceCollector
from agents.orchestrator.service import route_query

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
