from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from agents.library.agent import run_library_agent
from agents.library.api.schemas import LibraryChatRequest
from agents.library.models import BookRecord, GuideDocRecord


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
        queryUid="query-1",
        traceId="trace-1",
        conversationUid="conversation-1",
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
