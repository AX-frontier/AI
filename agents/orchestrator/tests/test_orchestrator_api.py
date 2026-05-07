from __future__ import annotations

from fastapi.testclient import TestClient

from agents.orchestrator.api.router import get_routing_evidence_collector
from agents.orchestrator.routing.evidence import AgentEvidence, RoutingEvidence
from app import app


class FixedEvidenceCollector:
    def collect(self, message: str) -> RoutingEvidence:
        return RoutingEvidence(
            main=AgentEvidence(score=0.83, reason="main fixed hit"),
            library=AgentEvidence(score=0.4, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )


def test_orchestrator_route_endpoint_returns_main_for_school_notice_query() -> None:
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector()
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_001",
            "traceId": "tr_001",
            "conversationUid": "conv_001",
            "message": "복수전공 신청 기간 알려줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "MAIN"
    assert payload["intent"] == "MAIN"
    assert payload["evidence"]["mainScore"] == 0.83
