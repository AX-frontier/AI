from __future__ import annotations

from agents.orchestrator.routing.evidence import AgentEvidence, RoutingEvidence
from agents.orchestrator.routing.router import EvidenceBasedRouter


def test_routes_to_document_review_when_document_score_is_high() -> None:
    decision = EvidenceBasedRouter().route(
        RoutingEvidence(
            main=AgentEvidence(score=0.9, reason="main hit"),
            library=AgentEvidence(score=0.9, reason="library hit"),
            document_review=AgentEvidence(score=0.75, reason="document hit"),
        )
    )

    assert decision.target_agent == "DOCUMENT_REVIEW"
    assert decision.intent == "DOCUMENT_REVIEW"


def test_routes_to_library_when_library_score_is_high() -> None:
    decision = EvidenceBasedRouter().route(
        RoutingEvidence(
            main=AgentEvidence(score=0.8, reason="main hit"),
            library=AgentEvidence(score=0.82, reason="library hit"),
            document_review=AgentEvidence(score=0.0, reason="no doc hit"),
        )
    )

    assert decision.target_agent == "LIBRARY"
    assert decision.intent == "LIBRARY"


def test_routes_to_main_when_only_main_vector_score_is_valid() -> None:
    decision = EvidenceBasedRouter().route(
        RoutingEvidence(
            main=AgentEvidence(score=0.62, reason="main hit"),
            library=AgentEvidence(score=0.4, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no doc hit"),
        )
    )

    assert decision.target_agent == "MAIN"
    assert decision.intent == "MAIN"


def test_routes_to_fallback_when_no_evidence_is_valid() -> None:
    decision = EvidenceBasedRouter().route(
        RoutingEvidence(
            main=AgentEvidence(score=0.2, reason="weak main"),
            library=AgentEvidence(score=0.4, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no doc hit"),
        )
    )

    assert decision.target_agent == "FALLBACK"
    assert decision.intent == "FALLBACK"
