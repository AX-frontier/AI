from __future__ import annotations

import pytest

from agents.main_agent.agent import run_main_agent
from agents.main_agent.api.schemas import MainChatRequest
from agents.main_agent.embedding import DeterministicEmbeddingProvider
from agents.main_agent.llm.mock import MockLLMClient
from agents.main_agent.models import MainChunkRecord
from agents.main_agent.retrieval import extract_main_search_keyword


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


@pytest.mark.parametrize(
    ("message", "keyword"),
    [
        ("ax프론티어에 대해 알고싶어", "ax 프론티어"),
        ("sw교육 관해서 궁금해", "sw 교육"),
        ("AI장학금 보고싶어", "ai 장학금"),
    ],
)
def test_main_search_keyword_normalizes_common_interest_phrases_and_mixed_tokens(
    message: str,
    keyword: str,
) -> None:
    assert extract_main_search_keyword(message) == keyword


def test_main_agent_handles_known_page_navigation_without_vector_search() -> None:
    repository = MockChunkRepository([])
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_page_001",
            traceId="tr_page_001",
            conversationUid="conv_page_001",
            message="학정관 페이지로 이동해줘",
        ),
        repository=repository,
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=MockLLMClient(),
    )

    assert response.targetAgent == "MAIN"
    assert response.fallbackUsed is False
    assert response.resultCount == 1
    assert "학술정보관 페이지" in response.answer
    assert "https://hsel.hansung.ac.kr/" in response.answer
    assert response.sources[0].url == "https://hsel.hansung.ac.kr/"
    assert repository.last_embedding is None


@pytest.mark.parametrize(
    "message",
    [
        "장학금 신청 방법 알려줘",
        "장학금 어떻게 신청해?",
        "장학 신청 절차 궁금해",
        "장학금 종류 알려줘",
    ],
)
def test_main_agent_answers_general_scholarship_guidance_without_notice_search(message: str) -> None:
    repository = MockChunkRepository(
        [
            MainChunkRecord(
                chunk_id="notice-scholarship-specific-0001",
                document_id="notice-scholarship-specific",
                text="국가장학금Ⅱ유형 동의서 제출 안내입니다.",
                score=0.95,
                metadata={
                    "title": "2025학년도 2학기 국가장학금Ⅱ유형(학업장려금) 동의서 제출 안내",
                    "category": "장학공지",
                    "url": "https://example.edu/specific-scholarship-notice",
                },
            )
        ]
    )

    response = run_main_agent(
        MainChatRequest(
            queryUid="q_scholarship_guide",
            traceId="tr_scholarship_guide",
            conversationUid="conv_scholarship_guide",
            message=message,
        ),
        repository=repository,
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=RecordingLLMClient("개별 공지 답변입니다."),
    )

    assert response.fallbackUsed is False
    assert response.intent == "MAIN_GENERAL"
    assert response.resultCount == 1
    assert response.sources[0].title == "장학금 안내"
    assert response.sources[0].url == "https://hansung.ac.kr/edubank/5762/subview.do"
    assert response.sources[0].category == "official-guide"
    assert "장학 종류와 학기별 공지에 따라 달라집니다" in response.answer
    assert "https://hansung.ac.kr/edubank/5762/subview.do" in response.answer
    assert "국가장학금Ⅱ유형" not in response.answer
    assert "https://example.edu/specific-scholarship-notice" not in response.answer
    assert repository.last_embedding is None


@pytest.mark.parametrize(
    "message",
    [
        "장학금 공지 알려줘",
        "국가장학금 마감 언제야",
        "주거안정장학금 지급 제외 기준",
    ],
)
def test_main_agent_keeps_specific_scholarship_notice_queries_on_rag_path(message: str) -> None:
    repository = MockChunkRepository(
        [
            MainChunkRecord(
                chunk_id="notice-scholarship-0001",
                document_id="notice-scholarship",
                text="주거안정장학금 지급 제외 기준과 국가장학금 마감 안내입니다.",
                score=0.93,
                metadata={
                    "title": "2026학년도 1학기 주거안정장학금 대학 자체 우선지원 및 지급 제외기준 안내",
                    "category": "장학공지",
                    "url": "https://example.edu/scholarship-notice",
                },
            )
        ]
    )

    response = run_main_agent(
        MainChatRequest(
            queryUid="q_scholarship_notice",
            traceId="tr_scholarship_notice",
            conversationUid="conv_scholarship_notice",
            message=message,
        ),
        repository=repository,
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=RecordingLLMClient("개별 장학 공지를 확인할 수 있습니다."),
    )

    assert response.fallbackUsed is False
    assert response.intent == "SCHOOL_NOTICE_QA"
    assert response.sources[0].url == "https://example.edu/scholarship-notice"
    assert "https://example.edu/scholarship-notice" in response.answer
    assert "https://hansung.ac.kr/edubank/5762/subview.do" not in response.answer


def test_main_agent_does_not_treat_library_hours_question_as_page_navigation() -> None:
    repository = MockChunkRepository([])
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_page_002",
            traceId="tr_page_002",
            conversationUid="conv_page_002",
            message="학정관 몇 시에 열어",
        ),
        repository=repository,
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=MockLLMClient(),
    )

    assert response.fallbackUsed is True
    assert response.sources == []
    assert repository.last_embedding is not None


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


def test_main_agent_filters_unrelated_links_from_answer_sources_and_prompt() -> None:
    llm_client = RecordingLLMClient("복수전공 신청 기간과 절차를 확인할 수 있습니다.")
    chunks = [
        MainChunkRecord(
            chunk_id="notice-major-0001",
            document_id="notice-major",
            text="복수전공 신청 기간과 변경 절차에 대한 학사 공지입니다.",
            score=0.91,
            metadata={
                "title": "복수전공 신청 안내",
                "category": "학사",
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
                "url": "https://example.edu/green-umbrella",
            },
        ),
    ]

    response = run_main_agent(
        MainChatRequest(
            queryUid="q_filter_links",
            traceId="tr_filter_links",
            conversationUid="conv_filter_links",
            message="복수전공 신청 기간 알려줘",
        ),
        repository=MockChunkRepository(chunks),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=llm_client,
    )

    assert response.fallbackUsed is False
    assert "복수전공 신청 안내" in response.answer
    assert "https://example.edu/major" in response.answer
    assert "Adobe 공동구매" not in response.answer
    assert "https://example.edu/adobe" not in response.answer
    assert "초록우산" not in response.answer
    assert "https://example.edu/green-umbrella" not in response.answer
    assert [source.url for source in response.sources] == ["https://example.edu/major"]
    assert len(llm_client.prompts) == 1
    prompt = llm_client.prompts[0]
    assert "url: https://example.edu/major" in prompt
    assert "https://example.edu/adobe" not in prompt
    assert "https://example.edu/green-umbrella" not in prompt


@pytest.mark.parametrize(
    "message",
    [
        "ax 프론티어 보고싶어",
        "ax프론티어에 대해 알고싶어",
        "AX프런티어 궁금해",
    ],
)
def test_main_agent_keeps_only_ax_frontier_link_for_frontier_query_alias(message: str) -> None:
    llm_client = RecordingLLMClient("AX 프런티어 챌린지 진행 상황을 확인할 수 있습니다.")
    chunks = [
        MainChunkRecord(
            chunk_id="notice-ax-frontier-0001",
            document_id="notice-ax-frontier",
            text="한성 AX 프런티어 챌린지 1단계 심사 통과 결과를 안내합니다.",
            score=0.93,
            metadata={
                "title": "제1회 한성 AX 프런티어 챌린지 — 1단계 심사 통과 결과 발표",
                "category": "한성공지",
                "url": "https://www.hansung.ac.kr/bbs/hansung/2127/221952/artclView.do",
            },
        ),
        MainChunkRecord(
            chunk_id="notice-gown-0001",
            document_id="notice-gown",
            text="학위수여식 관련 학사복 대여 안내입니다.",
            score=0.91,
            metadata={
                "title": "[총학생회] 2025학년도 전기 학위수여식 학사복 대여 안내",
                "category": "한성공지",
                "url": "https://www.hansung.ac.kr/bbs/hansung/2127/219632/artclView.do",
            },
        ),
        MainChunkRecord(
            chunk_id="notice-club-0001",
            document_id="notice-club",
            text="2025학년도 2학기 동아리활동 평가 결과 안내입니다.",
            score=0.9,
            metadata={
                "title": "2025학년도 2학기 동아리활동 평가 결과 안내",
                "category": "한성공지",
                "url": "https://www.hansung.ac.kr/bbs/hansung/2127/219691/artclView.do",
            },
        ),
    ]

    response = run_main_agent(
        MainChatRequest(
            queryUid="q_ax_frontier",
            traceId="tr_ax_frontier",
            conversationUid="conv_ax_frontier",
            message=message,
        ),
        repository=MockChunkRepository(chunks),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=llm_client,
    )

    assert response.fallbackUsed is False
    assert "제1회 한성 AX 프런티어 챌린지" in response.answer
    assert "https://www.hansung.ac.kr/bbs/hansung/2127/221952/artclView.do" in response.answer
    assert "학사복 대여" not in response.answer
    assert "동아리활동 평가" not in response.answer
    assert [source.url for source in response.sources] == [
        "https://www.hansung.ac.kr/bbs/hansung/2127/221952/artclView.do"
    ]
    assert len(llm_client.prompts) == 1
    prompt = llm_client.prompts[0]
    assert "url: https://www.hansung.ac.kr/bbs/hansung/2127/221952/artclView.do" in prompt
    assert "https://www.hansung.ac.kr/bbs/hansung/2127/219632/artclView.do" not in prompt
    assert "https://www.hansung.ac.kr/bbs/hansung/2127/219691/artclView.do" not in prompt


def test_main_agent_does_not_match_ax_inside_longer_latin_token() -> None:
    for message in ("ax 프론티어 보고싶어", "ax프론티어에 대해 알고싶어"):
        response = run_main_agent(
            MainChatRequest(
                queryUid="q_ax_fax",
                traceId="tr_ax_fax",
                conversationUid="conv_ax_fax",
                message=message,
            ),
            repository=MockChunkRepository(
                [
                    MainChunkRecord(
                        chunk_id="notice-fax-frontier-0001",
                        document_id="notice-fax-frontier",
                        text="FAX 프런티어 서류 제출 방식 안내입니다.",
                        score=0.93,
                        metadata={
                            "title": "FAX 프런티어 제출 안내",
                            "category": "한성공지",
                            "url": "https://example.edu/fax-frontier",
                        },
                    )
                ]
            ),
            embedding_provider=DeterministicEmbeddingProvider(),
            llm_client=RecordingLLMClient("무관한 답변입니다."),
        )

        assert response.fallbackUsed is True
        assert response.fallbackReasonCode == "TOPIC_MISMATCH_NO_DATA"
        assert response.sources == []
        assert "FAX 프런티어 제출 안내" not in response.answer
        assert "https://example.edu/fax-frontier" not in response.answer


def test_main_agent_applies_mixed_latin_korean_query_filter_to_general_topics() -> None:
    llm_client = RecordingLLMClient("SW 교육 프로그램 안내를 확인할 수 있습니다.")
    chunks = [
        MainChunkRecord(
            chunk_id="notice-sw-edu-0001",
            document_id="notice-sw-edu",
            text="SW 교육 프로그램 신청과 운영 일정 안내입니다.",
            score=0.93,
            metadata={
                "title": "SW 교육 프로그램 안내",
                "category": "한성공지",
                "url": "https://example.edu/sw-edu",
            },
        ),
        MainChunkRecord(
            chunk_id="notice-aws-edu-0001",
            document_id="notice-aws-edu",
            text="AWS 교육 특강 신청 안내입니다.",
            score=0.91,
            metadata={
                "title": "AWS 교육 특강 안내",
                "category": "한성공지",
                "url": "https://example.edu/aws-edu",
            },
        ),
    ]

    response = run_main_agent(
        MainChatRequest(
            queryUid="q_sw_edu",
            traceId="tr_sw_edu",
            conversationUid="conv_sw_edu",
            message="sw교육에 대해 알고싶어",
        ),
        repository=MockChunkRepository(chunks),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=llm_client,
    )

    assert response.fallbackUsed is False
    assert "SW 교육 프로그램 안내" in response.answer
    assert "https://example.edu/sw-edu" in response.answer
    assert "AWS 교육" not in response.answer
    assert "https://example.edu/aws-edu" not in response.answer
    assert [source.url for source in response.sources] == ["https://example.edu/sw-edu"]


def test_main_agent_removes_links_when_llm_description_says_unrelated() -> None:
    llm_client = RecordingLLMClient(
        "\n".join(
            [
                "1. AX 프런티어 챌린지 심사 결과를 확인할 수 있습니다.",
                "2. AX 프런티어와 직접 관련은 없으나 학사 행사 준비에 필요한 정보를 제공합니다.",
            ]
        )
    )
    chunks = [
        MainChunkRecord(
            chunk_id="notice-ax-frontier-0001",
            document_id="notice-ax-frontier",
            text="한성 AX 프런티어 챌린지 1단계 심사 통과 결과를 안내합니다.",
            score=0.93,
            metadata={
                "title": "제1회 한성 AX 프런티어 챌린지 — 1단계 심사 통과 결과 발표",
                "category": "한성공지",
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
                "url": "https://example.edu/gown",
            },
        ),
    ]

    response = run_main_agent(
        MainChatRequest(
            queryUid="q_ax_disclaimer",
            traceId="tr_ax_disclaimer",
            conversationUid="conv_ax_disclaimer",
            message="ax 프론티어 보고싶어",
        ),
        repository=MockChunkRepository(chunks),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=llm_client,
    )

    assert response.fallbackUsed is False
    assert "제1회 한성 AX 프런티어 챌린지" in response.answer
    assert "https://example.edu/ax-frontier" in response.answer
    assert "학사복 대여" not in response.answer
    assert "직접 관련은 없으나" not in response.answer
    assert "https://example.edu/gown" not in response.answer
    assert [source.url for source in response.sources] == ["https://example.edu/ax-frontier"]


def test_main_agent_falls_back_when_all_link_descriptions_are_unrelated() -> None:
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_ax_all_disclaimed",
            traceId="tr_ax_all_disclaimed",
            conversationUid="conv_ax_all_disclaimed",
            message="ax 프론티어 보고싶어",
        ),
        repository=MockChunkRepository(
            [
                MainChunkRecord(
                    chunk_id="notice-ax-frontier-gown-0001",
                    document_id="notice-ax-frontier-gown",
                    text="AX 프런티어와 직접 관련은 없으나 학위수여식 학사복 대여 안내입니다.",
                    score=0.93,
                    metadata={
                        "title": "AX 프런티어 참고 학사복 대여 안내",
                        "category": "한성공지",
                        "url": "https://example.edu/gown",
                    },
                )
            ]
        ),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=RecordingLLMClient("AX 프런티어와 직접 관련은 없으나 학사 행사 준비에 필요한 정보를 제공합니다."),
    )

    assert response.fallbackUsed is True
    assert response.fallbackReasonCode == "TOPIC_MISMATCH_NO_DATA"
    assert response.sources == []
    assert "학사복 대여" not in response.answer
    assert "https://example.edu/gown" not in response.answer


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


def test_main_agent_falls_back_when_ambiguous_boundary_results_are_not_scholarship_related() -> None:
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

    assert response.fallbackUsed is True
    assert response.answer.startswith("데이터 준비중입니다.")
    assert response.fallbackReasonCode == "TOPIC_MISMATCH_NO_DATA"
    assert response.sources == []
    assert "대신 참고하기 좋은 관련 공지를 먼저 추천드립니다." not in response.answer
    assert "공지 A" not in response.answer
    assert "https://example.edu/notice-a" not in response.answer
    assert llm_client.prompts == []


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
    assert [source.url for source in response.sources] == [
        "https://example.edu/notice-url-1",
        "https://example.edu/notice-url-2",
    ]


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


def test_main_agent_returns_fallback_for_topic_mismatch_query() -> None:
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_mismatch",
            traceId="tr_mismatch",
            conversationUid="conv_mismatch",
            message="우산 잃어버렸는데 어디에 전화해야되",
        ),
        repository=MockChunkRepository(
            [
                MainChunkRecord(
                    chunk_id="notice-umbrella-0001",
                    document_id="notice-umbrella",
                    text="초록우산 봉사단 모집 공지입니다.",
                    score=0.848,
                    metadata={
                        "title": "초록우산 봉사단 모집 안내",
                        "category": "한성공지",
                        "url": "https://example.edu/umbrella",
                    },
                ),
                MainChunkRecord(
                    chunk_id="notice-adobe-0001",
                    document_id="notice-adobe",
                    text="Adobe 공동구매 안내입니다.",
                    score=0.598,
                    metadata={
                        "title": "Adobe 공동구매 특별 프로모션",
                        "category": "한성공지",
                        "url": "https://example.edu/adobe",
                    },
                ),
            ]
        ),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=MockLLMClient("무관한 답변"),
    )

    assert response.fallbackUsed is True
    assert "직접적으로 일치하는 공지 데이터를 찾지 못했습니다" in response.fallbackReason
    assert response.fallbackReasonCode == "TOPIC_MISMATCH_NO_DATA"


def test_main_agent_returns_fallback_for_lost_item_query_without_lost_item_evidence() -> None:
    response = run_main_agent(
        MainChatRequest(
            queryUid="q_mismatch_lost_item",
            traceId="tr_mismatch_lost_item",
            conversationUid="conv_mismatch_lost_item",
            message="우산 잃어버렸는데 어디에 전화해야되?",
        ),
        repository=MockChunkRepository(
            [
                MainChunkRecord(
                    chunk_id="notice-umbrella-0001",
                    document_id="notice-umbrella",
                    text="초록우산 2026 대학생 봉사단 모집 공지입니다.",
                    score=0.9,
                    metadata={
                        "title": "[ESG센터] 초록우산 봉사단 모집",
                        "category": "한성공지",
                        "url": "https://example.edu/green-umbrella",
                    },
                ),
                MainChunkRecord(
                    chunk_id="notice-facility-0001",
                    document_id="notice-facility",
                    text="시설관리직 채용 문의는 아래 전화로 연락 바랍니다.",
                    score=0.82,
                    metadata={
                        "title": "시설관리직 채용공고",
                        "category": "한성공지",
                        "url": "https://example.edu/facility-hiring",
                    },
                ),
            ]
        ),
        embedding_provider=DeterministicEmbeddingProvider(),
        llm_client=MockLLMClient("무관한 답변"),
    )

    assert response.fallbackUsed is True
    assert response.fallbackReasonCode == "TOPIC_MISMATCH_NO_DATA"
