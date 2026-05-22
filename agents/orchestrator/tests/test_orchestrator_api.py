from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient

from agents.library.api.router import get_library_repository
from agents.library.models import BookRecord
from agents.main_agent.api.router import get_main_embedding_provider, get_main_llm_client
from agents.main_agent.embedding import get_embedding_provider
from agents.main_agent.embedding import DeterministicEmbeddingProvider
from agents.main_agent.llm.mock import MockLLMClient
from agents.main_agent.models import MainChunkRecord
from agents.main_agent.repository import get_main_chunk_repository
from agents.orchestrator.api.router import get_routing_evidence_collector
from agents.orchestrator.routing.evidence import AgentEvidence, RoutingEvidence
from agents.orchestrator.service import _ORCH_FOLLOWUP_MEMORY, _sse_event
from app import app

os.environ["MAIN_AGENT_EMBEDDING_PROVIDER"] = "deterministic"
os.environ["MAIN_AGENT_EMBEDDING_DIMENSIONS"] = "1536"
get_embedding_provider.cache_clear()


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


class OrchestratorMainMismatchRepository:
    def search_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MainChunkRecord]:
        return [
            MainChunkRecord(
                chunk_id="notice-umb-0001",
                document_id="notice-umb",
                text="초록우산 봉사단 모집 공지입니다.",
                score=0.848,
                metadata={
                    "title": "초록우산 봉사단 모집 안내",
                    "category": "한성공지",
                    "posted_date": "2026-03-01",
                    "url": "https://example.edu/umb-1",
                },
            ),
            MainChunkRecord(
                chunk_id="notice-umb-0002",
                document_id="notice-umb",
                text="Adobe 공동구매 안내입니다.",
                score=0.598,
                metadata={
                    "title": "Adobe 공동구매 특별 프로모션",
                    "category": "한성공지",
                    "posted_date": "2026-03-02",
                    "url": "https://example.edu/umb-2",
                },
            ),
        ]


class OrchestratorMainMixedRepository:
    def search_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MainChunkRecord]:
        return [
            MainChunkRecord(
                chunk_id="notice-major-0001",
                document_id="notice-major",
                text="복수전공 신청 기간과 변경 절차는 학사 공지를 확인해 주세요.",
                score=0.91,
                metadata={
                    "title": "복수전공 신청 안내",
                    "category": "학사",
                    "posted_date": "2026-03-01",
                    "url": "https://example.edu/major",
                },
            ),
            MainChunkRecord(
                chunk_id="notice-adobe-0001",
                document_id="notice-adobe",
                text="Adobe 공동구매 프로모션 안내입니다.",
                score=0.89,
                metadata={
                    "title": "Adobe 공동구매 특별 프로모션",
                    "category": "한성공지",
                    "posted_date": "2026-03-02",
                    "url": "https://example.edu/adobe",
                },
            ),
            MainChunkRecord(
                chunk_id="notice-volunteer-0001",
                document_id="notice-volunteer",
                text="초록우산 대학생 봉사단 모집 안내입니다.",
                score=0.88,
                metadata={
                    "title": "초록우산 봉사단 모집",
                    "category": "한성공지",
                    "posted_date": "2026-03-03",
                    "url": "https://example.edu/green-umbrella",
                },
            ),
        ]


class OrchestratorMainAxDisclaimedRepository:
    def search_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MainChunkRecord]:
        return [
            MainChunkRecord(
                chunk_id="notice-ax-frontier-0001",
                document_id="notice-ax-frontier",
                text="한성 AX 프런티어 챌린지 1단계 심사 통과 결과를 안내합니다.",
                score=0.93,
                metadata={
                    "title": "제1회 한성 AX 프런티어 챌린지 — 1단계 심사 통과 결과 발표",
                    "category": "한성공지",
                    "posted_date": "2026-03-01",
                    "url": "https://example.edu/ax-frontier",
                },
            ),
            MainChunkRecord(
                chunk_id="notice-ax-frontier-gown-0001",
                document_id="notice-ax-frontier-gown",
                text="AX 프런티어와 직접 관련은 없으나 학위수여식 학사복 대여 안내입니다.",
                score=0.91,
                metadata={
                    "title": "AX 프런티어 참고 학사복 대여 안내",
                    "category": "한성공지",
                    "posted_date": "2026-03-02",
                    "url": "https://example.edu/gown",
                },
            ),
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


def test_sse_event_preserves_korean_text_for_debuggability() -> None:
    event = _sse_event({"type": "chunk", "text": "문서 검토 결과입니다."})

    assert "문서 검토 결과" in event
    assert "\\ubb38" not in event


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


def test_orchestrator_route_endpoint_prefers_main_for_latest_notice_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.72, reason="main hit"),
            library=AgentEvidence(score=0.70, reason="library close hit"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_rule_main_001",
            "traceId": "tr_rule_main_001",
            "conversationUid": "conv_rule_main_001",
            "message": "오늘 새로 올라온 공지 있어?",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "MAIN"


def test_orchestrator_route_endpoint_prefers_library_for_library_hours_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.72, reason="main hit"),
            library=AgentEvidence(score=0.70, reason="library close hit"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_rule_lib_001",
            "traceId": "tr_rule_lib_001",
            "conversationUid": "conv_rule_lib_001",
            "message": "학술정보관 오늘 몇 시까지?",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "LIBRARY"


def test_orchestrator_route_endpoint_sends_known_page_navigation_to_main() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.2, reason="weak main"),
            library=AgentEvidence(score=0.95, reason="strong library keyword"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_page_nav_001",
            "traceId": "tr_page_nav_001",
            "conversationUid": "conv_page_nav_001",
            "message": "학술정보관 페이지로 이동해줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "MAIN"
    assert payload["intent"] == "MAIN"


def test_orchestrator_route_endpoint_routes_main_when_library_domain_terms_are_absent() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.35, reason="main moderate hit"),
            library=AgentEvidence(score=0.9, reason="library high but domain mismatch"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_rule_main_umb_001",
            "traceId": "tr_rule_main_umb_001",
            "conversationUid": "conv_rule_main_umb_001",
            "message": "우산 잃어버렸는데 어디에 전화해야되",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "MAIN"


def test_orchestrator_route_endpoint_returns_fallback_for_ambiguous_low_margin() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.83, reason="main close hit"),
            library=AgentEvidence(score=0.79, reason="library close hit"),
            document_review=AgentEvidence(score=0.05, reason="weak document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_amb_001",
            "traceId": "tr_amb_001",
            "conversationUid": "conv_amb_001",
            "message": "링크 정리해줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "FALLBACK"
    assert payload["intent"] == "FALLBACK"
    assert payload["routingMode"] == "FRESH"
    assert payload["routingReasonCode"] == "AMBIGUOUS_LOW_MARGIN"
    assert payload["confidence"] == 0.83
    assert "모호" in payload["reason"]
    assert "학교공지 안내" in payload["reason"]
    assert "도서 검색" in payload["reason"]
    assert "문서 검토" in payload["reason"]
    assert payload["evidence"]["mainScore"] == 0.83
    assert payload["evidence"]["libraryScore"] == 0.79


def test_orchestrator_route_endpoint_keeps_main_for_low_top1_when_not_ambiguous_enough() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.598, reason="main weak hit"),
            library=AgentEvidence(score=0.0, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="weak document"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_amb_003",
            "traceId": "tr_amb_003",
            "conversationUid": "conv_amb_003",
            "message": "복수전공 신청 기간 알려줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "MAIN"
    assert payload["routingReasonCode"] == "FRESH_DEFAULT"


def test_orchestrator_route_endpoint_keeps_original_route_when_margin_is_not_low() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.83, reason="main hit"),
            library=AgentEvidence(score=0.74, reason="library hit"),
            document_review=AgentEvidence(score=0.05, reason="weak document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_amb_002",
            "traceId": "tr_amb_002",
            "conversationUid": "conv_amb_002",
            "message": "복수전공 신청 기간 알려줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "MAIN"
    assert payload["intent"] == "MAIN"
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


def test_orchestrator_route_endpoint_routes_library_for_loan_extension_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.55, reason="main weak hit"),
            library=AgentEvidence(score=0.9, reason="library guide hit with loan extension"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/route",
        json={
            "queryUid": "q_002_loan",
            "traceId": "tr_002_loan",
            "conversationUid": "conv_001",
            "message": "학술정보관 이용 안내로 대출 연장 방법 알려줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "LIBRARY"
    assert payload["intent"] == "LIBRARY"


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


def test_orchestrator_route_endpoint_keeps_main_for_unrelated_low_confidence_query() -> None:
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
    assert payload["targetAgent"] == "MAIN"
    assert payload["intent"] == "MAIN"
    assert payload["routingReasonCode"] == "FRESH_DEFAULT"
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


def test_orchestrator_chat_endpoint_filters_unrelated_main_sources_after_routing() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.9, reason="main hit"),
            library=AgentEvidence(score=0.2, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    app.dependency_overrides[get_main_chunk_repository] = lambda: OrchestratorMainMixedRepository()
    app.dependency_overrides[get_main_embedding_provider] = lambda: DeterministicEmbeddingProvider()
    app.dependency_overrides[get_main_llm_client] = lambda: MockLLMClient("복수전공 신청 안내입니다.")
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_main_filter_001",
            "traceId": "tr_main_filter_001",
            "conversationUid": "conv_main_filter_001",
            "message": "복수전공 신청 기간 알려줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "MAIN"
    assert [source["url"] for source in payload["sources"]] == ["https://example.edu/major"]
    assert "https://example.edu/adobe" not in payload["answer"]
    assert "https://example.edu/green-umbrella" not in payload["answer"]


def test_orchestrator_chat_endpoint_returns_data_preparing_when_main_has_no_relevant_data() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.84, reason="main hit"),
            library=AgentEvidence(score=0.1, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    app.dependency_overrides[get_main_chunk_repository] = lambda: OrchestratorMainMismatchRepository()
    app.dependency_overrides[get_main_embedding_provider] = lambda: DeterministicEmbeddingProvider()
    app.dependency_overrides[get_main_llm_client] = lambda: MockLLMClient("무관한 답변")
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_data_prep_001",
            "traceId": "tr_data_prep_001",
            "conversationUid": "conv_data_prep_001",
            "message": "우산 잃어버렸는데 어디에 전화해야되?",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "DATA_PREPARING"
    assert payload["intent"] == "DATA_PREPARING"
    assert payload["fallbackUsed"] is True
    assert payload["confidence"] == 0.0
    assert "데이터가 아직 준비되지 않았습니다" in payload["answer"]


def test_orchestrator_chat_stream_endpoint_returns_data_preparing_when_main_has_no_relevant_data() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.84, reason="main hit"),
            library=AgentEvidence(score=0.1, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    app.dependency_overrides[get_main_chunk_repository] = lambda: OrchestratorMainMismatchRepository()
    app.dependency_overrides[get_main_embedding_provider] = lambda: DeterministicEmbeddingProvider()
    app.dependency_overrides[get_main_llm_client] = lambda: MockLLMClient("무관한 답변")
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat/stream",
        json={
            "queryUid": "q_data_prep_stream_001",
            "traceId": "tr_data_prep_stream_001",
            "conversationUid": "conv_data_prep_stream_001",
            "message": "우산 잃어버렸는데 어디에 전화해야되?",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    events = [line for line in response.text.splitlines() if line.startswith("data: ")]
    done_payload = json.loads(events[-1].removeprefix("data: "))
    assert done_payload["type"] == "done"
    assert done_payload["targetAgent"] == "DATA_PREPARING"
    assert done_payload["intent"] == "DATA_PREPARING"
    assert "데이터가 아직 준비되지 않았습니다" in done_payload["answer"]


def test_orchestrator_chat_stream_endpoint_filters_unrelated_main_sources_after_routing() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.9, reason="main hit"),
            library=AgentEvidence(score=0.2, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    app.dependency_overrides[get_main_chunk_repository] = lambda: OrchestratorMainMixedRepository()
    app.dependency_overrides[get_main_embedding_provider] = lambda: DeterministicEmbeddingProvider()
    app.dependency_overrides[get_main_llm_client] = lambda: MockLLMClient("복수전공 신청 안내입니다.")
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat/stream",
        json={
            "queryUid": "q_main_filter_stream_001",
            "traceId": "tr_main_filter_stream_001",
            "conversationUid": "conv_main_filter_stream_001",
            "message": "복수전공 신청 기간 알려줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    events = [line for line in response.text.splitlines() if line.startswith("data: ")]
    done_payload = json.loads(events[-1].removeprefix("data: "))
    assert done_payload["type"] == "done"
    assert done_payload["targetAgent"] == "MAIN"
    assert [source["url"] for source in done_payload["sources"]] == ["https://example.edu/major"]
    assert "https://example.edu/adobe" not in done_payload["answer"]
    assert "https://example.edu/green-umbrella" not in done_payload["answer"]


def test_orchestrator_chat_stream_endpoint_does_not_emit_disclaimed_unrelated_links() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.9, reason="main hit"),
            library=AgentEvidence(score=0.2, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    app.dependency_overrides[get_main_chunk_repository] = lambda: OrchestratorMainAxDisclaimedRepository()
    app.dependency_overrides[get_main_embedding_provider] = lambda: DeterministicEmbeddingProvider()
    app.dependency_overrides[get_main_llm_client] = lambda: MockLLMClient(
        "\n".join(
            [
                "1. AX 프런티어 챌린지 심사 결과를 확인할 수 있습니다.",
                "2. AX 프런티어와 직접 관련은 없으나 학사 행사 준비에 필요한 정보를 제공합니다.",
            ]
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat/stream",
        json={
            "queryUid": "q_ax_disclaimed_stream_001",
            "traceId": "tr_ax_disclaimed_stream_001",
            "conversationUid": "conv_ax_disclaimed_stream_001",
            "message": "ax 프론티어 보고싶어",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")]
    emitted_text = "\n".join(str(event.get("text", "")) + "\n" + str(event.get("answer", "")) for event in events)
    done_payload = events[-1]
    assert done_payload["type"] == "done"
    assert done_payload["targetAgent"] == "MAIN"
    assert [source["url"] for source in done_payload["sources"]] == ["https://example.edu/ax-frontier"]
    assert "https://example.edu/gown" not in emitted_text
    assert "직접 관련은 없으나" not in emitted_text
    assert "학사복 대여" not in emitted_text


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


def test_orchestrator_chat_endpoint_executes_library_agent_for_loan_extension_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.4, reason="weak main"),
            library=AgentEvidence(score=0.9, reason="library guide hit with loan extension"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    app.dependency_overrides[get_library_repository] = lambda: OrchestratorLibraryMockRepository()
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_library_loan_001",
            "traceId": "tr_library_loan_001",
            "conversationUid": "conv_001",
            "message": "학술정보관 이용 안내로 대출 연장 방법 알려줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "LIBRARY"

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


def test_orchestrator_chat_endpoint_returns_fallback_for_unrelated_ambiguous_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.2, reason="weak main"),
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
    assert payload["intent"] == "FALLBACK"
    assert payload["fallbackUsed"] is True


def test_orchestrator_chat_endpoint_returns_fallback_for_ambiguous_low_margin() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.83, reason="main close hit"),
            library=AgentEvidence(score=0.79, reason="library close hit"),
            document_review=AgentEvidence(score=0.05, reason="weak document hit"),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_amb_chat_001",
            "traceId": "tr_amb_chat_001",
            "conversationUid": "conv_amb_chat_001",
            "message": "이거 처리해줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "FALLBACK"
    assert payload["intent"] == "FALLBACK"
    assert payload["fallbackUsed"] is True
    assert payload["confidence"] == 0.0
    assert "모호" in payload["answer"]
    assert "학교공지 안내" in payload["answer"]
    assert "도서 검색" in payload["answer"]
    assert "문서 검토" in payload["answer"]


def test_orchestrator_chat_endpoint_returns_data_preparing_when_main_sources_do_not_match_vague_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.9, reason="main hit"),
            library=AgentEvidence(score=0.1, reason="weak library"),
            document_review=AgentEvidence(score=0.0, reason="weak document"),
        )
    )
    app.dependency_overrides[get_main_chunk_repository] = lambda: OrchestratorMainMockRepository()
    app.dependency_overrides[get_main_embedding_provider] = lambda: DeterministicEmbeddingProvider()
    app.dependency_overrides[get_main_llm_client] = lambda: MockLLMClient("메인 에이전트 응답입니다.")
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_amb_chat_002",
            "traceId": "tr_amb_chat_002",
            "conversationUid": "conv_amb_chat_002",
            "message": "이거 처리해줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "DATA_PREPARING"
    assert payload["intent"] == "DATA_PREPARING"
    assert payload["fallbackUsed"] is True
    assert payload["sources"] == []


def test_orchestrator_chat_endpoint_keeps_library_for_explicit_book_query_even_with_low_top1() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.3, reason="weak main"),
            library=AgentEvidence(score=0.6, reason="library book hit"),
            document_review=AgentEvidence(score=0.0, reason="weak document"),
        )
    )
    app.dependency_overrides[get_library_repository] = lambda: OrchestratorLibraryMockRepository()
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_lib_keep_001",
            "traceId": "tr_lib_keep_001",
            "conversationUid": "conv_lib_keep_001",
            "message": "도서관에 AI관련 책이 있어?",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "LIBRARY"
    assert payload["intent"] in {"BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION"}
    assert payload["fallbackUsed"] is False


def test_orchestrator_chat_endpoint_keeps_library_when_main_and_library_are_both_high() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.968, reason="main high hit"),
            library=AgentEvidence(score=0.98, reason="library high hit"),
            document_review=AgentEvidence(score=0.0, reason="weak document"),
        )
    )
    app.dependency_overrides[get_library_repository] = lambda: OrchestratorLibraryMockRepository()
    client = TestClient(app)

    response = client.post(
        "/orchestrator/chat",
        json={
            "queryUid": "q_lib_keep_002",
            "traceId": "tr_lib_keep_002",
            "conversationUid": "conv_lib_keep_002",
            "message": "도서관에 AI관련 책이 있어?",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "LIBRARY"
    assert payload["fallbackUsed"] is False


def test_route_recomputes_and_switches_library_to_main_on_new_query() -> None:
    _ORCH_FOLLOWUP_MEMORY.clear()
    app.dependency_overrides[get_routing_evidence_collector] = lambda: FixedEvidenceCollector(
        RoutingEvidence(
            main=AgentEvidence(score=0.2, reason="weak main"),
            library=AgentEvidence(score=0.9, reason="library hit"),
            document_review=AgentEvidence(score=0.0, reason="no document hit"),
        )
    )
    app.dependency_overrides[get_library_repository] = lambda: OrchestratorLibraryMockRepository()
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
