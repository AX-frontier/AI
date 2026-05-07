from __future__ import annotations

from fastapi.testclient import TestClient

from agents.library.api.router import get_library_repository
from agents.library.models import BookRecord
from app import app


class ApiMockRepository:
    def search_books(self, keyword: str, limit: int = 5) -> list[BookRecord]:
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
