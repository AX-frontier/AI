from __future__ import annotations

from agents.main_agent.agent import run_main_agent
from agents.main_agent.api.schemas import MainChatRequest
from agents.main_agent.models import MainChunkRecord


class MockChunkRepository:
    def __init__(self, chunks: list[MainChunkRecord]):
        self.chunks = chunks
        self.last_embedding: list[float] | None = None

    def search_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MainChunkRecord]:
        self.last_embedding = query_embedding
        return self.chunks[:limit]


def test_main_agent_builds_answer_with_sources() -> None:
    repository = MockChunkRepository(
        [
            MainChunkRecord(
                chunk_id="notice-219610-0001",
                document_id="219610",
                text="2026학년도 1학기 복수·부전공 신청 및 변경에 관한 사항입니다.",
                score=0.88,
                metadata={
                    "title": "2026학년도 1학기 복수·부전공 신청 및 변경신청 안내",
                    "category": "학사",
                    "department": "학사운영팀",
                    "posted_date": "2026-01-26",
                    "url": "https://www.hansung.ac.kr/bbs/hansung/2127/219610/artclView.do",
                },
            )
        ]
    )
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_001",
            traceId="tr_001",
            conversationUid="conv_001",
            message="복수전공 신청 기간 알려줘",
        ),
        repository=repository,
    )

    assert response.targetAgent == "MAIN"
    assert response.intent == "ACADEMIC_INFO_QA"
    assert response.fallbackUsed is False
    assert response.resultCount == 1
    assert response.sources[0].documentId == "219610"
    assert repository.last_embedding is not None


def test_main_agent_returns_fallback_when_chunks_are_empty() -> None:
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_002",
            traceId="tr_002",
            conversationUid="conv_002",
            message="없는 공지 알려줘",
        ),
        repository=MockChunkRepository([]),
    )

    assert response.targetAgent == "MAIN"
    assert response.fallbackUsed is True
    assert response.resultCount == 0
    assert response.sources == []
