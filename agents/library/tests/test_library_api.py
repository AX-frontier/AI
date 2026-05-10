from __future__ import annotations

from fastapi.testclient import TestClient

from agents.library.api.router import get_library_repository
from agents.library.models import BookRecord
from app import app


class ApiMockRepository:
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


class BookLocationMockRepository:
    """'클린 코드 책 어디 있어?' 시나리오용 — 제목 기반 서브스트링 매칭."""

    _BOOK = BookRecord(
        id=3,
        bib_no="BIB-003",
        reg_no="REG-003",
        title="클린 코드",
        author="로버트 마틴",
        publisher="인사이트",
        publish_year=2013,
        holding_call_no="005.133 ㅁ123ㅋ",
        material_type="단행본",
        location_symbol="LIB",
        stack_location="제1자료실",
        stack_shelf="A-05",
    )

    def search_books(
        self,
        keyword: str,
        limit: int = 5,
        *,
        location_question: bool = False,
    ) -> list[BookRecord]:
        if keyword.lower() in self._BOOK.title.lower():
            return [self._BOOK]
        return []

    def search_guide_docs(self, keyword: str, limit: int = 3) -> list:
        return []


class QualifiedKeywordMockRepository:
    """필드 설명어가 검색어에 남으면 결과가 나오지 않는 시나리오."""

    _BOOKS = {
        "슈카친구들": BookRecord(
            id=4,
            bib_no="BIB-004",
            reg_no="REG-004",
            title="돈 Economy now",
            author="슈카친구들",
            publisher="슈카친구들",
            publish_year=2026,
            holding_call_no="001 ㅅ658ㄷ",
            material_type="단행본",
            location_symbol="LIB",
            stack_location="인문자연과학자료실",
            stack_shelf="1-A-1-d",
        ),
        "Facet": BookRecord(
            id=5,
            bib_no="BIB-005",
            reg_no="REG-005",
            title="Dewey Decimal Classification",
            author="Mohinder Partap Satija",
            publisher="Facetpublishing",
            publish_year=2013,
            holding_call_no="024.41 S253a",
            material_type="단행본",
            location_symbol="LIB",
            stack_location="인문자연과학자료실",
            stack_shelf="7-A-3-b",
        ),
        "001 ㄱ785ㄷ": BookRecord(
            id=6,
            bib_no="BIB-006",
            reg_no="REG-006",
            title="도넛을 구멍만 남기고 먹는 방법",
            author="오사카대학 쇼세키카 프로젝트",
            publisher="글항아리",
            publish_year=2014,
            holding_call_no="001 ㄱ785ㄷ",
            material_type="단행본",
            location_symbol="LIB",
            stack_location="인문자연과학자료실",
            stack_shelf="1-A-1-c",
        ),
        "1-A-1-d": BookRecord(
            id=7,
            bib_no="BIB-007",
            reg_no="REG-007",
            title="대탈주",
            author="연구모임 사회비판과대안",
            publisher="사월의책",
            publish_year=2014,
            holding_call_no="301.52 ㄹ624ㄷ",
            material_type="단행본",
            location_symbol="LIB",
            stack_location="사회과학자료실",
            stack_shelf="1-A-1-d",
        ),
    }

    def search_books(
        self,
        keyword: str,
        limit: int = 5,
        *,
        location_question: bool = False,
    ) -> list[BookRecord]:
        book = self._BOOKS.get(keyword)
        return [book] if book else []

    def search_guide_docs(self, keyword: str, limit: int = 3) -> list:
        return []


def test_library_chat_endpoint_returns_contract_payload() -> None:
    app.dependency_overrides[get_library_repository] = lambda: ApiMockRepository()
    client = TestClient(app)

    response = client.post(
        "/library/chat",
        json={
            "queryUid": "11111111-1111-4111-8111-111111111111",
            "traceId": "22222222-2222-4222-8222-222222222222",
            "conversationUid": "33333333-3333-4333-8333-333333333333",
            "message": "파이썬 도서 검색",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "LIBRARY"
    assert payload["intent"] == "BOOK_SEARCH"
    assert payload["searchKeyword"] == "파이썬"
    assert payload["resultCount"] == 1
    assert payload["matchedBooks"][0]["bibNo"] == "BIB-001"


def test_book_location_with_book_suffix_returns_matched_book() -> None:
    """BE 이슈 재현: '클린 코드 책 어디 있어?' → resultCount=1, fallbackUsed=false."""
    app.dependency_overrides[get_library_repository] = lambda: BookLocationMockRepository()
    client = TestClient(app)

    response = client.post(
        "/library/chat",
        json={
            "queryUid": "11111111-1111-4111-8111-111111111111",
            "traceId": "22222222-2222-4222-8222-222222222222",
            "conversationUid": "33333333-3333-4333-8333-333333333333",
            "message": "클린 코드 책 어디 있어?",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "BOOK_LOCATION"
    assert payload["searchKeyword"] == "클린 코드"
    assert payload["resultCount"] == 1
    assert payload["matchedBooks"][0]["title"] == "클린 코드"
    assert payload["fallbackUsed"] is False


def test_book_search_strips_field_qualifiers_before_repository_lookup() -> None:
    app.dependency_overrides[get_library_repository] = lambda: QualifiedKeywordMockRepository()
    client = TestClient(app)

    cases = [
        ("슈카친구들 저자 도서 검색", "슈카친구들"),
        ("Facet 출판사 도서 검색", "Facet"),
        ("001 ㄱ785ㄷ 청구기호 위치 알려줘", "001 ㄱ785ㄷ"),
        ("1-A-1-d 서가 위치 알려줘", "1-A-1-d"),
    ]

    for message, expected_keyword in cases:
        response = client.post(
            "/library/chat",
            json={
                "queryUid": "11111111-1111-4111-8111-111111111111",
                "traceId": "22222222-2222-4222-8222-222222222222",
                "conversationUid": "33333333-3333-4333-8333-333333333333",
                "message": message,
            },
        )
        payload = response.json()
        assert response.status_code == 200
        assert payload["searchKeyword"] == expected_keyword
        assert payload["resultCount"] == 1
        assert payload["fallbackUsed"] is False

    app.dependency_overrides.clear()


def test_library_chat_endpoint_rejects_non_uuid_identifiers() -> None:
    app.dependency_overrides[get_library_repository] = lambda: ApiMockRepository()
    client = TestClient(app)

    response = client.post(
        "/library/chat",
        json={
            "queryUid": "query-1",
            "traceId": "tr-20260507-0001",
            "conversationUid": "conv-20260507-0001",
            "message": "파이썬 도서 검색",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 422
