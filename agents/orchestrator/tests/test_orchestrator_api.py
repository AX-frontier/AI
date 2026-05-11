from __future__ import annotations

from fastapi.testclient import TestClient

from agents.orchestrator.api.router import get_routing_evidence_collector
from agents.orchestrator.routing.evidence import AgentEvidence, RoutingEvidence
from app import app


class FixedEvidenceCollector:
    def __init__(self, evidence: RoutingEvidence):
        self.evidence = evidence

    def collect(self, message: str) -> RoutingEvidence:
        return self.evidence


def test_orchestrator_route_endpoint_returns_main_for_school_notice_query() -> None:
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.83, reason="main fixed hit"),
            library=AgentEvidence(score=0.4, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
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
    assert payload["queryUid"] == "q_001"
    assert payload["traceId"] == "tr_001"
    assert payload["conversationUid"] == "conv_001"
    assert payload["evidence"]["mainScore"] == 0.83


def test_orchestrator_route_endpoint_returns_library_for_library_query() -> None:
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.55, reason="main weak hit"),
            library=AgentEvidence(score=0.82, reason="library guide hit"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_002",
            "traceId": "tr_002",
            "conversationUid": "conv_001",
            "message": "도서관 운영 시간 알려줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "LIBRARY"
    assert payload["intent"] == "LIBRARY"
    assert payload["evidence"]["libraryScore"] == 0.82


def test_orchestrator_route_endpoint_returns_document_review_for_review_query() -> None:
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.52, reason="main weak hit"),
            library=AgentEvidence(score=0.4, reason="weak library"),
            document_review=AgentEvidence(score=0.75, reason="document review hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_003",
            "traceId": "tr_003",
            "conversationUid": "conv_001",
            "message": "이 공문 문장 검토해줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "DOCUMENT_REVIEW"
    assert payload["intent"] == "DOCUMENT_REVIEW"
    assert payload["evidence"]["documentReviewScore"] == 0.75


def test_orchestrator_route_endpoint_prioritizes_explicit_document_review() -> None:
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.862, reason="main weak hit"),
            library=AgentEvidence(score=0.98, reason="library false positive"),
            document_review=AgentEvidence(score=0.85, reason="document review hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_005",
            "traceId": "tr_005",
            "conversationUid": "conv_001",
            "message": "기안할 문서가 있는데 검토해줄 수 있어?",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "DOCUMENT_REVIEW"
    assert payload["intent"] == "DOCUMENT_REVIEW"
    assert payload["evidence"]["documentReviewScore"] == 0.85


def test_orchestrator_route_endpoint_returns_fallback_for_unrelated_query() -> None:
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.2, reason="weak main"),
            library=AgentEvidence(score=0.4, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_004",
            "traceId": "tr_004",
            "conversationUid": "conv_001",
            "message": "안녕",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "FALLBACK"
    assert payload["intent"] == "FALLBACK"
    assert "evidence" in payload
