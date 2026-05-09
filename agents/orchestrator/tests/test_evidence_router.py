from __future__ import annotations

from agents.main_agent.models import MainChunkRecord
from agents.orchestrator.routing.evidence import AgentEvidence, RoutingEvidence, RoutingEvidenceCollector
from agents.orchestrator.routing.router import EvidenceBasedRouter


class FixedEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class RecordingMainRepository:
    def __init__(self, chunks: list[MainChunkRecord]):
        self.chunks = chunks
        self.similar_limits: list[int] = []
        self.keyword_limits: list[int] = []

    def search_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MainChunkRecord]:
        self.similar_limits.append(limit)
        return self.chunks[:limit]

    def search_keyword_chunks(self, terms: list[str], *, limit: int = 5) -> list[MainChunkRecord]:
        self.keyword_limits.append(limit)
        return []


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


def test_main_evidence_uses_main_retriever_reranked_top_hits() -> None:
    repository = RecordingMainRepository(
        [
            MainChunkRecord(
                chunk_id="notice-general-0001",
                document_id="notice-general",
                text="일반 공지입니다.",
                score=0.7,
                metadata={"title": "일반 공지", "category": "공지"},
            ),
            MainChunkRecord(
                chunk_id="notice-major-0001",
                document_id="notice-major",
                text="복수전공 신청 기간 및 변경 절차 안내입니다.",
                score=0.62,
                metadata={"title": "복수전공 신청 기간 안내", "category": "학사"},
            ),
            MainChunkRecord(
                chunk_id="notice-third-0001",
                document_id="notice-third",
                text="복수전공 관련 참고 안내입니다.",
                score=0.6,
                metadata={"title": "복수전공 참고 안내", "category": "학사"},
            ),
            MainChunkRecord(
                chunk_id="notice-fourth-0001",
                document_id="notice-fourth",
                text="복수전공 관련 추가 안내입니다.",
                score=0.59,
                metadata={"title": "복수전공 추가 안내", "category": "학사"},
            ),
        ]
    )

    evidence = RoutingEvidenceCollector(
        main_repository=repository,
        embedding_provider=FixedEmbeddingProvider(),
    )._collect_main_evidence("복수전공 신청 기간 알려줘")

    assert repository.similar_limits == [20]
    assert repository.keyword_limits == [12]
    assert evidence.score > 0.7
    assert evidence.reason.startswith("main reranked hits: 복수전공 신청 기간 안내")
    assert "notice-major-0001" in evidence.reason
    assert "notice-third-0001" in evidence.reason
    assert "notice-fourth-0001" in evidence.reason
    assert "notice-general-0001" not in evidence.reason


def test_main_evidence_returns_zero_when_main_retriever_has_no_hits() -> None:
    repository = RecordingMainRepository([])

    evidence = RoutingEvidenceCollector(
        main_repository=repository,
        embedding_provider=FixedEmbeddingProvider(),
    )._collect_main_evidence("없는 공지 알려줘")

    assert repository.similar_limits == [20]
    assert evidence.score == 0.0
    assert evidence.reason == "no main vector chunk hit"
