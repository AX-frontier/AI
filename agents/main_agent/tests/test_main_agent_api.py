from __future__ import annotations

from fastapi.testclient import TestClient

from agents.main_agent.models import MainChunkRecord
from agents.main_agent.repository import get_main_chunk_repository
from app import app


class ApiMockChunkRepository:
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
                text="수강신청 정정 기간은 학사 공지에서 확인할 수 있습니다.",
                score=0.82,
                metadata={
                    "title": "수강신청 정정 안내",
                    "category": "학사",
                    "posted_date": "2026-03-01",
                    "url": "https://example.edu/notice-1",
                },
            )
        ]


def test_main_chat_endpoint_returns_spring_compatible_payload() -> None:
    app.dependency_overrides[get_main_chunk_repository] = lambda: ApiMockChunkRepository()
    client = TestClient(app)

    response = client.post(
        "/main/chat",
        json={
            "queryUid": "query-1",
            "traceId": "trace-1",
            "conversationUid": "conversation-1",
            "message": "수강신청 정정 기간 알려줘",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "MAIN"
    assert payload["intent"] == "ACADEMIC_INFO_QA"
    assert payload["fallbackUsed"] is False
    assert payload["sources"][0]["chunkId"] == "notice-1-0001"
