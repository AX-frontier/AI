from __future__ import annotations

import pytest

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


class RecordingLLMClient:
    def __init__(self, answer: str = "검색 근거 기반 답변입니다."):
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


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
    assert '"복수전공 신청 기간"와 관련해 확인할 수 있는 공식 링크를 찾았습니다.' in response.answer
    assert "2026학년도 1학기 복수·부전공 신청 및 변경신청 안내" in response.answer
    assert "복수·부전공 신청 안내에 대한 답변입니다." in response.answer
    assert "https://www.hansung.ac.kr/bbs/hansung/2127/219610/artclView.do" in response.answer
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
    assert response.answer.startswith('"복수전공 신청 기간"와 관련해 확인할 수 있는 공식 링크를 찾았습니다.')
    assert "검색 결과에서 관련도가 가장 높은 대표 링크 1건을 안내드립니다." in response.answer
    assert "https://www.hansung.ac.kr/bbs/hansung/2127/219610/artclView.do" in response.answer


def test_main_agent_prompt_limits_reference_urls_to_top_three() -> None:
    llm_client = RecordingLLMClient(
        "\n".join(
            [
                "1. 첫 번째 공지에서 확인할 수 있습니다.",
                "2. 두 번째 공지에서 확인할 수 있습니다.",
                "3. 세 번째 공지에서 확인할 수 있습니다.",
            ]
        )
    )

    chunks = [
        MainChunkRecord(
            chunk_id=f"notice-{index}-0001",
            document_id=f"notice-{index}",
            text=f"공지 {index} 본문입니다.",
            score=0.9 - (index * 0.01),
            metadata={
                "title": f"공지 {index}",
                "category": "학사",
                "url": f"https://example.edu/notice-{index}",
            },
        )
        for index in range(1, 5)
    ]

    response = run_main_agent(
        MainChatRequest(
            queryUid="q_prompt",
            traceId="tr_prompt",
            conversationUid="conv_prompt",
            message="공지 URL 알려줘",
        ),
        repository=MockChunkRepository(chunks),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=llm_client,
    )

    assert "첫 번째 공지에서 확인할 수 있습니다." in response.answer
    assert "두 번째 공지에서 확인할 수 있습니다." in response.answer
    assert "세 번째 공지에서 확인할 수 있습니다." in response.answer
    assert "1. 공지 1" in response.answer
    assert "2. 공지 2" in response.answer
    assert "3. 공지 3" in response.answer
    assert "https://example.edu/notice-1" in response.answer
    assert "https://example.edu/notice-2" in response.answer
    assert "https://example.edu/notice-3" in response.answer
    assert "https://example.edu/notice-4" not in response.answer
    assert len(llm_client.prompts) == 1
    prompt = llm_client.prompts[0]
    assert "URL은 절대 출력하지 마세요." in prompt
    assert "검색 source:" in prompt
    assert "title: 공지 1" in prompt
    assert "title: 공지 2" in prompt
    assert "title: 공지 3" in prompt
    assert "title: 공지 4" not in prompt
    assert "url: https://example.edu/notice-1" in prompt
    assert "https://example.edu/notice-4" not in prompt
    assert response.sources[0].url == "https://example.edu/notice-1"


def test_main_agent_shows_single_link_when_boundary_is_clear_and_top1_is_high() -> None:
    llm_client = RecordingLLMClient("핵심 내용과 대상, 일정, 유의사항을 확인할 수 있습니다.")
    chunks = [
        MainChunkRecord(
            chunk_id="notice-a-0001",
            document_id="notice-a",
            text="대표 공지 본문입니다.",
            score=0.95,
            metadata={
                "title": "대표 공지",
                "category": "학사",
                "url": "https://example.edu/notice-a",
            },
        ),
        MainChunkRecord(
            chunk_id="notice-b-0001",
            document_id="notice-b",
            text="보조 공지 본문입니다.",
            score=0.50,
            metadata={
                "title": "보조 공지",
                "category": "학사",
                "url": "https://example.edu/notice-b",
            },
        ),
    ]

    response = run_main_agent(
        MainChatRequest(
            queryUid="q_single",
            traceId="tr_single",
            conversationUid="conv_single",
            message="대표 공지 알려줘",
        ),
        repository=MockChunkRepository(chunks),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=llm_client,
    )

    assert "대표 링크 1건" in response.answer
    assert "1. 대표 공지" in response.answer
    assert "2. " not in response.answer
    assert "https://example.edu/notice-b" not in response.answer
    assert len(llm_client.prompts) == 1
    assert "한국어 3~4문장" in llm_client.prompts[0]


def test_main_agent_keeps_three_links_when_mixed_threshold_is_not_met() -> None:
    llm_client = RecordingLLMClient(
        "\n".join(
            [
                "1. 첫 번째 공지에서 확인할 수 있습니다.",
                "2. 두 번째 공지에서 확인할 수 있습니다.",
                "3. 세 번째 공지에서 확인할 수 있습니다.",
            ]
        )
    )
    chunks = [
        MainChunkRecord(
            chunk_id=f"notice-mixed-{index}-0001",
            document_id=f"notice-mixed-{index}",
            text=f"공지 {index} 본문입니다.",
            score=score,
            metadata={
                "title": f"공지 {index}",
                "category": "학사",
                "url": f"https://example.edu/notice-mixed-{index}",
            },
        )
        for index, score in enumerate([0.90, 0.82, 0.81], start=1)
    ]

    response = run_main_agent(
        MainChatRequest(
            queryUid="q_mixed",
            traceId="tr_mixed",
            conversationUid="conv_mixed",
            message="공지 알려줘",
        ),
        repository=MockChunkRepository(chunks),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=llm_client,
    )

    assert "1. 공지 1" in response.answer
    assert "2. 공지 2" in response.answer
    assert "3. 공지 3" in response.answer


def test_main_agent_keeps_three_links_when_boundary_is_ambiguous_even_if_top1_is_high() -> None:
    llm_client = RecordingLLMClient(
        "\n".join(
            [
                "1. 첫 번째 공지에서 확인할 수 있습니다.",
                "2. 두 번째 공지에서 확인할 수 있습니다.",
                "3. 세 번째 공지에서 확인할 수 있습니다.",
            ]
        )
    )
    chunks = [
        MainChunkRecord(
            chunk_id="notice-a-0001",
            document_id="notice-a",
            text="대표 공지 본문입니다.",
            score=0.90,
            metadata={
                "title": "공지 A",
                "category": "학사",
                "url": "https://example.edu/notice-a",
            },
        ),
        MainChunkRecord(
            chunk_id="notice-b-0001",
            document_id="notice-b",
            text="차순위 공지 본문입니다.",
            score=0.88,
            metadata={
                "title": "공지 B",
                "category": "학사",
                "url": "https://example.edu/notice-b",
            },
        ),
        MainChunkRecord(
            chunk_id="notice-c-0001",
            document_id="notice-c",
            text="세 번째 공지 본문입니다.",
            score=0.85,
            metadata={
                "title": "공지 C",
                "category": "학사",
                "url": "https://example.edu/notice-c",
            },
        ),
    ]

    response = run_main_agent(
        MainChatRequest(
            queryUid="q_ambiguous_boundary",
            traceId="tr_ambiguous_boundary",
            conversationUid="conv_ambiguous_boundary",
            message="장학금 공지 알려줘",
        ),
        repository=MockChunkRepository(chunks),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=llm_client,
    )

    assert "1. 공지 A" in response.answer
    assert "2. 공지 B" in response.answer
    assert "3. 공지 C" in response.answer


def test_main_agent_template_answer_omits_empty_urls_from_reference_list() -> None:
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_template",
            traceId="tr_template",
            conversationUid="conv_template",
            message="공지 URL 알려줘",
        ),
        repository=MockChunkRepository(
            [
                MainChunkRecord(
                    chunk_id="notice-url-1-0001",
                    document_id="notice-url-1",
                    text="URL이 있는 공지입니다.",
                    score=0.9,
                    metadata={
                        "title": "URL 있는 공지",
                        "category": "학사",
                        "url": "https://example.edu/notice-url-1",
                    },
                ),
                MainChunkRecord(
                    chunk_id="notice-url-empty-0001",
                    document_id="notice-url-empty",
                    text="URL이 없는 공지입니다.",
                    score=0.88,
                    metadata={
                        "title": "URL 없는 공지",
                        "category": "학사",
                    },
                ),
                MainChunkRecord(
                    chunk_id="notice-url-2-0001",
                    document_id="notice-url-2",
                    text="두 번째 URL 공지입니다.",
                    score=0.86,
                    metadata={
                        "title": "두 번째 URL 공지",
                        "category": "학사",
                        "url": "https://example.edu/notice-url-2",
                    },
                ),
            ]
        ),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=MockLLMClient(should_fail=True),
    )

    assert "1. URL 있는 공지" in response.answer
    assert "https://example.edu/notice-url-1" in response.answer
    assert "2. 두 번째 URL 공지" in response.answer
    assert "https://example.edu/notice-url-2" in response.answer
    assert "URL 없는 공지와 관련된 한성대학교 공식 안내 링크입니다." not in response.answer
    assert "URL 없는 공지" in response.sources[1].title


@pytest.mark.parametrize(
    ("message", "title", "text"),
    [
        (
            "복수전공 신청 기간 알려줘",
            "2026학년도 1학기 복수·부전공 신청 및 변경신청 안내",
            "복수전공과 부전공 신청 기간 및 변경 절차 안내입니다.",
        ),
        (
            "수강신청 정정 기간 언제야?",
            "2026학년도 1학기 수강신청 정정 안내",
            "수강신청 정정 기간과 신청 방법에 대한 학사 공지입니다.",
        ),
        (
            "휴복학 신청 기간 알려줘",
            "2026학년도 1학기 휴·복학 신청 안내",
            "휴학과 복학 신청 기간 및 처리 절차 안내입니다.",
        ),
    ],
)
def test_main_agent_keeps_top_source_for_search_quality_baseline(
    message: str,
    title: str,
    text: str,
) -> None:
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_quality",
            traceId="tr_quality",
            conversationUid="conv_quality",
            message=message,
        ),
        repository=MockChunkRepository(
            [
                MainChunkRecord(
                    chunk_id="notice-quality-0001",
                    document_id="notice-quality",
                    text=text,
                    score=0.9,
                    metadata={
                        "title": title,
                        "category": "학사",
                        "posted_date": "2026-01-26",
                        "url": "https://example.edu/quality",
                    },
                )
            ]
        ),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=MockLLMClient("검색 기준 답변입니다."),
    )

    assert response.fallbackUsed is False
    assert response.sources[0].title == title
    assert response.sources[0].score >= 0.9
