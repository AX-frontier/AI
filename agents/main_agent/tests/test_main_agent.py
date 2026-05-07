from __future__ import annotations

from agents.main_agent.agent import run_main_agent
from agents.main_agent.api.schemas import MainChatRequest
from agents.main_agent.embedding import DeterministicEmbeddingProvider
from agents.main_agent.llm.mock import MockLLMClient
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
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=MockLLMClient("복수·부전공 신청 안내에 대한 답변입니다."),
    )

    assert response.targetAgent == "MAIN"
    assert response.intent == "ACADEMIC_INFO_QA"
    assert response.fallbackUsed is False
    assert response.answer == "복수·부전공 신청 안내에 대한 답변입니다."
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
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=MockLLMClient(),
    )

    assert response.targetAgent == "MAIN"
    assert response.fallbackUsed is True
    assert response.resultCount == 0
    assert response.sources == []


def test_main_agent_returns_fallback_when_top_score_is_low() -> None:
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_003",
            traceId="tr_003",
            conversationUid="conv_003",
            message="관련 낮은 공지 알려줘",
        ),
        repository=MockChunkRepository(
            [
                MainChunkRecord(
                    chunk_id="notice-low-0001",
                    document_id="notice-low",
                    text="관련성이 낮은 공지입니다.",
                    score=0.2,
                    metadata={
                        "title": "관련성이 낮은 공지",
                        "category": "학사",
                        "url": "https://example.edu/low",
                    },
                )
            ]
        ),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=MockLLMClient(),
    )

    assert response.targetAgent == "MAIN"
    assert response.fallbackUsed is True
    assert response.fallbackReason == "관련 학교 공지 또는 안내 chunk의 유사도가 낮습니다."


def test_main_agent_falls_back_to_template_when_llm_fails() -> None:
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_004",
            traceId="tr_004",
            conversationUid="conv_004",
            message="복수전공 신청 기간 알려줘",
        ),
        repository=MockChunkRepository(
            [
                MainChunkRecord(
                    chunk_id="notice-219610-0001",
                    document_id="219610",
                    text="2026학년도 1학기 복수·부전공 신청 및 변경에 관한 사항입니다.",
                    score=0.88,
                    metadata={
                        "title": "2026학년도 1학기 복수·부전공 신청 및 변경신청 안내",
                        "category": "학사",
                        "posted_date": "2026-01-26",
                        "url": "https://www.hansung.ac.kr/bbs/hansung/2127/219610/artclView.do",
                    },
                )
            ]
        ),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=MockLLMClient(should_fail=True),
    )

    assert response.fallbackUsed is False
    assert response.answer.startswith("2026학년도 1학기 복수·부전공 신청")
