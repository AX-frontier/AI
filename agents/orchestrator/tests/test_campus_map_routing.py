from __future__ import annotations

from agents.orchestrator.routing.evidence import AgentEvidence, RoutingEvidence
from agents.orchestrator.routing.router import EvidenceBasedRouter


def test_routes_to_campus_map_for_building_location_query() -> None:
    decision = EvidenceBasedRouter().route(
        RoutingEvidence(
            main=AgentEvidence(score=0.1, reason="weak main"),
            library=AgentEvidence(score=0.5, reason="library mention"),
            document_review=AgentEvidence(score=0.0, reason="no doc"),
            campus_map=AgentEvidence(score=0.76, reason="상상관 location hit"),
        ),
        message="상상관 위치 알려줘",
    )

    assert decision.target_agent == "CAMPUS_MAP"


def test_book_location_query_does_not_route_to_campus_map() -> None:
    decision = EvidenceBasedRouter().route(
        RoutingEvidence(
            main=AgentEvidence(score=0.1, reason="weak main"),
            library=AgentEvidence(score=0.71, reason="book hit"),
            document_review=AgentEvidence(score=0.0, reason="no doc"),
            campus_map=AgentEvidence(score=0.75, reason="generic location"),
        ),
        message="클린 코드 책 위치 알려줘",
    )

    assert decision.target_agent == "LIBRARY"
