from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from agents.library.agent import decide_library_llm_policy, run_library_agent
from agents.library.api.schemas import LibraryChatRequest
from agents.library.classifier import IntentClassification
from agents.library.models import BookRecord, GuideChunkRecord, GuideDocRecord
from agents.library.retrieval import GuideRetriever, GuideSearchResult
from agents.main_agent.llm.mock import MockLLMClient


class RecordingLLMClient:
    def __init__(self, answer: str = "LLM 안내 답변입니다."):
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


@dataclass
class MockLibraryRepository:
    books: list[BookRecord]
    guides: list[GuideDocRecord]

    def search_books(self, keyword: str, limit: int = 5) -> list[BookRecord]:
        normalized = keyword.lower()
        results = []
        for book in self.books:
            fields = [
                book.title,
                book.author,
                book.publisher,
                book.holding_call_no,
                book.stack_location,
                book.stack_shelf,
            ]
            if any(normalized in (field or "").lower() for field in fields):
                results.append(book)
        return results[:limit]

    def search_guide_docs(self, keyword: str, limit: int = 3) -> list[GuideDocRecord]:
        normalized = keyword.lower()
        results = [
            guide
            for guide in self.guides
            if normalized in guide.title.lower() or normalized in guide.content.lower()
        ]
        return results[:limit]


def make_request(message: str) -> LibraryChatRequest:
    return LibraryChatRequest(
        queryUid="11111111-1111-4111-8111-111111111111",
        traceId="22222222-2222-4222-8222-222222222222",
        conversationUid="33333333-3333-4333-8333-333333333333",
        message=message,
    )


def make_repository() -> MockLibraryRepository:
    return MockLibraryRepository(
        books=[
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
            ),
            BookRecord(
                id=2,
                bib_no="BIB-002",
                reg_no="REG-002",
                title="데이터베이스 시스템",
                author="박데이터",
                publisher="생능출판",
                publish_year=2022,
                holding_call_no="005.74 ㅂ234ㄷ",
                material_type="단행본",
                location_symbol="LIB",
                stack_location="제2자료실",
                stack_shelf="B-03",
            ),
        ],
        guides=[
            GuideDocRecord(
                id=10,
                source_url="https://library.example.edu/hours",
                title="학술정보관 운영 시간",
                content="학술정보관은 학기 중 평일 09:00부터 21:00까지 운영합니다.",
                updated_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            )
        ],
    )


def test_book_title_search_returns_matched_books() -> None:
    response = run_library_agent(make_request("파이썬 도서 검색"), make_repository())

    assert response.targetAgent == "LIBRARY"
    assert response.intent == "BOOK_SEARCH"
    assert response.searchKeyword == "파이썬"
    assert response.resultCount == 1
    assert response.matchedBooks[0].title == "파이썬 자료구조"


def test_author_publisher_call_no_stack_location_and_shelf_search_work() -> None:
    repo = make_repository()

    cases = [
        ("김코딩 도서 검색", "파이썬 자료구조"),
        ("생능출판 도서 검색", "데이터베이스 시스템"),
        ("005.74 도서 검색", "데이터베이스 시스템"),
        ("제1자료실 도서 검색", "파이썬 자료구조"),
        ("B-03 도서 검색", "데이터베이스 시스템"),
    ]

    for message, expected_title in cases:
        response = run_library_agent(make_request(message), repo)
        assert response.resultCount == 1
        assert response.matchedBooks[0].title == expected_title


def test_book_location_answer_uses_location_fields() -> None:
    response = run_library_agent(make_request("파이썬 자료구조 위치 어디 있어?"), make_repository())

    assert response.intent == "BOOK_LOCATION"
    assert response.resultCount == 1
    assert "제1자료실" in response.answer
    assert response.matchedBooks[0].holdingCallNo == "005.133 ㄱ123ㅍ"


def test_guide_question_uses_guide_docs() -> None:
    response = run_library_agent(make_request("학술정보관 운영 시간 알려줘"), make_repository())

    assert response.intent == "LIBRARY_GUIDE"
    assert response.fallbackUsed is False
    assert response.sources[0].title == "학술정보관 운영 시간"
    assert "09:00" in response.answer


def test_guide_question_uses_injected_llm_answer() -> None:
    response = run_library_agent(
        make_request("학술정보관 운영 시간 알려줘"),
        make_repository(),
        llm_client=MockLLMClient("학술정보관은 평일 09:00부터 21:00까지 운영합니다."),
    )

    assert response.intent == "LIBRARY_GUIDE"
    assert response.fallbackUsed is False
    assert response.answer == "학술정보관은 평일 09:00부터 21:00까지 운영합니다."


def test_guide_llm_failure_falls_back_to_chunk_summary() -> None:
    response = run_library_agent(
        make_request("학술정보관 운영 시간 알려줘"),
        make_repository(),
        llm_client=MockLLMClient(should_fail=True),
    )

    assert response.intent == "LIBRARY_GUIDE"
    assert response.fallbackUsed is False
    assert response.answer.startswith("학술정보관 운영 시간")
    assert "09:00" in response.answer


def test_guide_llm_prompt_contains_question_context_and_source() -> None:
    llm_client = RecordingLLMClient()

    response = run_library_agent(
        make_request("학술정보관 운영 시간 알려줘"),
        make_repository(),
        llm_client=llm_client,
    )

    assert response.answer == "LLM 안내 답변입니다."
    assert len(llm_client.prompts) == 1
    prompt = llm_client.prompts[0]
    assert "사용자 질문: 운영 시간" in prompt
    assert "학술정보관 운영 시간" in prompt
    assert "09:00부터 21:00" in prompt
    assert "https://library.example.edu/hours" in prompt


def test_book_search_ignores_injected_llm_client() -> None:
    llm_client = RecordingLLMClient("이 답변이 나오면 안 됩니다.")
    response = run_library_agent(
        make_request("파이썬 도서 검색"),
        make_repository(),
        llm_client=llm_client,
    )

    assert response.intent == "BOOK_SEARCH"
    assert response.fallbackUsed is False
    assert response.answer != "이 답변이 나오면 안 됩니다."
    assert "파이썬 자료구조" in response.answer
    assert llm_client.prompts == []


def test_guide_without_search_result_does_not_call_llm() -> None:
    llm_client = RecordingLLMClient()

    response = run_library_agent(
        make_request("도서관 외계어 안내 알려줘"),
        MockLibraryRepository(books=[], guides=[]),
        llm_client=llm_client,
    )

    assert response.fallbackUsed is True
    assert llm_client.prompts == []


def test_llm_policy_disables_answer_llm_for_book_intents() -> None:
    classification = IntentClassification(
        intent="BOOK_SEARCH",
        confidence=0.93,
        reason="book hit",
        evidence={},
    )

    decision = decide_library_llm_policy(classification)

    assert decision.use_answer_llm is False
    assert decision.use_intent_llm is False
    assert decision.reason == "book_intent_uses_structured_template"


def test_llm_policy_enables_answer_llm_for_guide_chunks() -> None:
    classification = IntentClassification(
        intent="LIBRARY_GUIDE",
        confidence=0.88,
        reason="guide hit",
        evidence={},
    )
    guide_result = GuideSearchResult(
        keyword="운영 시간",
        docs=[
            GuideDocRecord(
                id=10,
                source_url="https://library.example.edu/hours",
                title="학술정보관 운영 시간",
                content="학술정보관은 학기 중 평일 09:00부터 21:00까지 운영합니다.",
            )
        ],
        chunks=[
            GuideChunkRecord(
                id=101,
                guide_doc_id=10,
                title="학술정보관 운영 시간",
                source_url="https://library.example.edu/hours",
                content="학술정보관은 학기 중 평일 09:00부터 21:00까지 운영합니다.",
                chunk_index=0,
                score=0.9,
            )
        ],
    )

    decision = decide_library_llm_policy(classification, guide_result)

    assert decision.use_answer_llm is True
    assert decision.use_intent_llm is False
    assert decision.reason == "guide_search_result_available_for_rag_answer"


def test_llm_policy_marks_ambiguous_intent_candidate_without_answer_llm() -> None:
    classification = IntentClassification(
        intent="LIBRARY_GENERAL",
        confidence=0.4,
        reason="ambiguous",
        evidence={},
        ambiguous=True,
    )

    decision = decide_library_llm_policy(classification)

    assert decision.use_answer_llm is False
    assert decision.use_intent_llm is True
    assert decision.reason == "no_guide_search_result"


def test_semantic_guide_hit_answers_when_exact_keyword_does_not_match(monkeypatch) -> None:
    class FakeEmbeddingProvider:
        dimensions = 384

        def embed_query(self, text: str) -> list[float]:
            assert "오늘 몇 시" in text
            return [0.1] * 384

        def embed_document(self, text: str) -> list[float]:
            return [0.1] * 384

    class SemanticGuideRepository:
        def search_books(self, keyword: str, limit: int = 5) -> list[BookRecord]:
            return []

        def search_guide_docs(self, keyword: str, limit: int = 3) -> list[GuideDocRecord]:
            return []

        def search_guide_chunks_by_keyword(
            self,
            keyword: str,
            *,
            limit: int = 12,
        ) -> list[GuideChunkRecord]:
            return []

        def search_guide_chunks_by_embedding(
            self,
            query_embedding: list[float],
            *,
            limit: int = 12,
            min_score: float = 0.35,
        ) -> list[GuideChunkRecord]:
            assert len(query_embedding) == 384
            return [
                GuideChunkRecord(
                    id=101,
                    guide_doc_id=10,
                    title="학술정보관 운영 시간",
                    source_url="https://library.example.edu/hours",
                    content="학술정보관은 학기 중 평일 09:00부터 21:00까지 운영합니다.",
                    chunk_index=0,
                    score=0.82,
                    updated_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
                )
            ]

    monkeypatch.setattr(
        "agents.library.retrieval.get_embedding_provider",
        lambda: FakeEmbeddingProvider(),
    )

    response = run_library_agent(make_request("오늘 몇 시까지 열어요?"), SemanticGuideRepository())

    assert response.intent == "LIBRARY_GUIDE"
    assert response.fallbackUsed is False
    assert response.sources[0].title == "학술정보관 운영 시간"
    assert "09:00" in response.answer


def test_guide_retriever_dedupes_keyword_and_semantic_chunks_by_score() -> None:
    class FakeEmbeddingProvider:
        dimensions = 384

        def embed_query(self, text: str) -> list[float]:
            return [0.2] * 384

        def embed_document(self, text: str) -> list[float]:
            return [0.2] * 384

    class DuplicateChunkRepository:
        def search_guide_docs(self, keyword: str, limit: int = 3) -> list[GuideDocRecord]:
            return []

        def search_guide_chunks_by_keyword(
            self,
            keyword: str,
            *,
            limit: int = 12,
        ) -> list[GuideChunkRecord]:
            return [
                GuideChunkRecord(
                    id=101,
                    guide_doc_id=10,
                    title="반납 안내",
                    source_url="https://library.example.edu/return",
                    content="키워드 검색으로 찾은 낮은 점수 chunk",
                    chunk_index=0,
                    score=0.55,
                )
            ]

        def search_guide_chunks_by_embedding(
            self,
            query_embedding: list[float],
            *,
            limit: int = 12,
            min_score: float = 0.35,
        ) -> list[GuideChunkRecord]:
            return [
                GuideChunkRecord(
                    id=101,
                    guide_doc_id=10,
                    title="반납 안내",
                    source_url="https://library.example.edu/return",
                    content="의미 검색으로 찾은 높은 점수 chunk",
                    chunk_index=0,
                    score=0.88,
                ),
                GuideChunkRecord(
                    id=102,
                    guide_doc_id=10,
                    title="반납 안내",
                    source_url="https://library.example.edu/return",
                    content="같은 문서의 다른 chunk",
                    chunk_index=1,
                    score=0.86,
                ),
            ]

    result = GuideRetriever(
        DuplicateChunkRepository(),
        embedding_provider=FakeEmbeddingProvider(),
    ).retrieve("책 늦게 반납", limit=3)

    assert len(result.chunks) == 1
    assert result.chunks[0].id == 101
    assert result.chunks[0].score > 0.88
    assert result.docs[0].title == "반납 안내"


def test_library_word_does_not_force_book_search_for_guide_question() -> None:
    response = run_library_agent(make_request("도서관 운영 시간 알려줘"), make_repository())

    assert response.intent == "LIBRARY_GUIDE"
    assert response.fallbackUsed is False


def test_no_search_results_returns_fallback() -> None:
    response = run_library_agent(make_request("없는책 도서 검색"), make_repository())

    assert response.fallbackUsed is True
    assert response.fallbackReason == "검색 조건과 일치하는 도서를 찾지 못했습니다."
    assert response.resultCount == 0
    assert response.matchedBooks == []


def test_response_field_names_match_spring_contract() -> None:
    response = run_library_agent(make_request("파이썬 도서 검색"), make_repository())
    payload = response.model_dump(mode="json")

    assert set(payload) == {
        "targetAgent",
        "intent",
        "answer",
        "sources",
        "confidence",
        "fallbackUsed",
        "fallbackReason",
        "searchKeyword",
        "resultCount",
        "matchedBooks",
    }
    assert set(payload["matchedBooks"][0]) == {
        "id",
        "bibNo",
        "regNo",
        "title",
        "author",
        "publisher",
        "publishYear",
        "holdingCallNo",
        "materialType",
        "locationSymbol",
        "stackLocation",
        "stackShelf",
    }
