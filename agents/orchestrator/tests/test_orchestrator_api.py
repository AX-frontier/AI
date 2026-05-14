from __future__ import annotations

from fastapi.testclient import TestClient

from agents.library.api.router import get_library_repository
from agents.library.models import BookRecord
from agents.main_agent.api.router import get_main_embedding_provider, get_main_llm_client
from agents.main_agent.embedding import DeterministicEmbeddingProvider
from agents.main_agent.llm.mock import MockLLMClient
from agents.main_agent.models import MainChunkRecord
from agents.main_agent.repository import get_main_chunk_repository
from agents.orchestrator.api.router import get_routing_evidence_collector
from agents.orchestrator.routing.evidence import AgentEvidence, RoutingEvidence
from agents.orchestrator.service import _ORCH_FOLLOWUP_MEMORY
from app import app


class FixedEvidenceCollector:
    def __init__(self, evidence: RoutingEvidence):
        self.evidence = evidence

    def collect(self, message: str) -> RoutingEvidence:
        return self.evidence


class OrchestratorMainMockRepository:
    def search_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MainChunkRecord]:
        return [
            MainChunkRecord(
                chunk_id="notice-1-0001",
                document_id="notice-1",
                text="복수전공 신청 기간은 학사 공지를 확인해 주세요.",
                score=0.88,
                metadata={
                    "title": "복수전공 신청 안내",
                    "category": "학사",
                    "posted_date": "2026-03-01",
                    "url": "https://example.edu/main-1",
                },
            )
        ]


class OrchestratorLibraryMockRepository:
    def search_books(
        self,
        keyword: str,
        limit: int = 5,
        *,
        location_question: bool = False,
    ) -> list[BookRecord]:
        return [
            BookRecord(
                id=1,
                bib_no="BIB-001",
                reg_no="REG-001",
                title="파이썬 자료구조",
                author="김코딩",
                publisher="한빛미디어",
                publish_year=2024,
                holding_call_no="005.133 ㄱ123ㅍ",
                material_type="단행본",
                location_symbol="LIB",
                stack_location="제1자료실",
                stack_shelf="A-12",
            )
        ]

    def search_guide_docs(self, keyword: str, limit: int = 3) -> list:
        return []

    def search_guide_chunks_by_keyword(
        self,
        keyword: str,
        *,
        limit: int = 12,
    ) -> list:
        return []

    def search_guide_chunks_by_embedding(
        self,
        query_embedding: list[float],
        *,
        limit: int = 12,
        min_score: float = 0.35,
    ) -> list:
        return []


def test_orchestrator_route_endpoint_returns_main_for_school_notice_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
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
    assert payload["routingMode"] == "FRESH"
    assert payload["routingReasonCode"] == "FRESH_DEFAULT"


def test_orchestrator_route_endpoint_returns_library_for_library_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
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
    assert payload["routingMode"] == "FRESH"


def test_orchestrator_route_endpoint_returns_document_review_for_review_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
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
    _ORCH_FOLLOWUP_MEMORY.clear()
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
    _ORCH_FOLLOWUP_MEMORY.clear()
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


def test_orchestrator_chat_endpoint_executes_main_agent() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.9, reason="main hit"),
            library=AgentEvidence(score=0.2, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    app.dependency_overrides[get_main_chunk_repository] = lambda: OrchestratorMainMockRepository()
    app.dependency_overrides[get_main_embedding_provider] = lambda: DeterministicEmbeddingProvider()
    app.dependency_overrides[get_main_llm_client] = lambda: MockLLMClient("메인 에이전트 응답입니다.")
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_main_001",
            "traceId": "tr_main_001",
            "conversationUid": "conv_001",
            "message": "복수전공 신청 기간 알려줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "MAIN"
    assert payload["intent"] == "ACADEMIC_INFO_QA"
    assert "메인 에이전트 응답입니다." in payload["answer"]
    assert "https://example.edu/main-1" in payload["answer"]


def test_orchestrator_chat_endpoint_executes_library_agent() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.3, reason="weak main"),
            library=AgentEvidence(score=0.9, reason="library hit"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    app.dependency_overrides[get_library_repository] = lambda: OrchestratorLibraryMockRepository()
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_library_001",
            "traceId": "tr_library_001",
            "conversationUid": "conv_001",
            "message": "파이썬 책 어디 있어?",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "LIBRARY"
    assert payload["intent"] == "BOOK_LOCATION"
    assert payload["matchedBooks"][0]["title"] == "파이썬 자료구조"


def test_orchestrator_chat_endpoint_executes_document_review_agent() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.2, reason="weak main"),
            library=AgentEvidence(score=0.1, reason="weak library"),
            document_review=AgentEvidence(score=0.9, reason="document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_doc_001",
            "traceId": "tr_doc_001",
            "conversationUid": "conv_001",
            "message": "기안할 문서가 있는데 검토해줄 수 있어?",
            "document": {
                "title": "예시 공문",
                "docType": "OFFICIAL_DOCUMENT",
                "bodyText": "2026-05-04 10:26:39\n붙임 1. 안내문 1부.\n끝.",
            },
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "DOCUMENT_REVIEW"
    assert payload["intent"] == "DOCUMENT_REVIEW"
    assert payload["status"] == "COMPLETED"


def test_orchestrator_chat_endpoint_requests_document_input_when_document_body_missing() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.2, reason="weak main"),
            library=AgentEvidence(score=0.1, reason="weak library"),
            document_review=AgentEvidence(score=0.9, reason="document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_doc_002",
            "traceId": "tr_doc_002",
            "conversationUid": "conv_001",
            "message": "기안할 문서가 있는데 검토해줄 수 있어?",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "DOCUMENT_REVIEW"
    assert payload["intent"] == "DOCUMENT_REVIEW_REQUIRED"
    assert payload["requiresDocumentInput"] is True
    assert payload["fallbackUsed"] is False
    assert "문서 본문" in payload["answer"]


def test_orchestrator_chat_endpoint_returns_fallback_for_unrelated_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.2, reason="weak main"),
            library=AgentEvidence(score=0.2, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_fallback_001",
            "traceId": "tr_fallback_001",
            "conversationUid": "conv_001",
            "message": "안녕",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "FALLBACK"
    assert payload["fallbackUsed"] is True


def test_route_recomputes_and_switches_library_to_main_on_new_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.2, reason="weak main"),
            library=AgentEvidence(score=0.9, reason="library hit"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    client = TestClient(app)

    first = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_switch_001",
            "traceId": "tr_switch_001",
            "conversationUid": "conv_switch",
            "message": "파이썬 책 어디 있어?",
        },
    )
    assert first.status_code == 200
    assert first.json()["targetAgent"] == "LIBRARY"

    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.92, reason="main hit"),
            library=AgentEvidence(score=0.3, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    app.dependency_overrides[get_main_chunk_repository] = lambda: OrchestratorMainMockRepository()
    app.dependency_overrides[get_main_embedding_provider] = lambda: DeterministicEmbeddingProvider()
    app.dependency_overrides[get_main_llm_client] = lambda: MockLLMClient("메인 에이전트 응답입니다.")

    second = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_switch_002",
            "traceId": "tr_switch_002",
            "conversationUid": "conv_switch",
            "message": "복수전공 신청 기간 알려줘",
        },
    )

    app.dependency_overrides.clear()
    assert second.status_code == 200
    assert second.json()["targetAgent"] == "MAIN"


def test_route_endpoint_uses_followup_sticky_tiebreak() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    _ORCH_FOLLOWUP_MEMORY["conv_followup"] = {
        "targetAgent": "MAIN",
        "topic": "복수전공 신청 기간",
        "updatedAt": "2099-01-01T00:00:00+00:00",
    }
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.72, reason="main close hit"),
            library=AgentEvidence(score=0.79, reason="library close hit"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    client = TestClient(app)
    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_followup_001",
            "traceId": "tr_followup_001",
            "conversationUid": "conv_followup",
            "message": "그럼 링크도 줘",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "MAIN"
    assert payload["routingMode"] == "FOLLOWUP_STICKY"
    assert payload["routingReasonCode"] == "FOLLOWUP_REUSE"
