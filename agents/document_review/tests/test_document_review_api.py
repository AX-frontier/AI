from __future__ import annotations

from fastapi.testclient import TestClient

from app import app


def test_document_review_chat_api_returns_workspace_payload() -> None:
    client = TestClient(app)
    response = client.post(
        "/document-review/chat",
        json={
            "queryUid": "00000000-0000-0000-0000-000000000001",
            "traceId": "00000000-0000-0000-0000-000000000002",
            "conversationUid": "00000000-0000-0000-0000-000000000003",
            "message": "전자결재 문서를 검토해줘",
            "document": {
                "title": "예시 공문",
                "docType": "OFFICIAL_DOCUMENT",
                "bodyText": "2026-05-04 10:26:39\n붙임 1. 안내문 1부.\n끝.",
                "bodyHtml": "<p>본문</p><table><tr><th>회계연도</th><th>소요예산</th></tr><tr><td>2026학년도</td><td>254,400</td></tr></table>",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["targetAgent"] == "DOCUMENT_REVIEW"
    assert payload["intent"] == "DOCUMENT_REVIEW"
    assert payload["answer"]
    assert payload["sources"] == []
    assert payload["findings"]
    assert payload["extractedTables"][0]["rows"][0] == ["회계연도", "소요예산"]
    assert payload["revisedDocument"]["content"]
    assert "<table>" in payload["revisedDocument"]["htmlContent"]
    assert payload["reviewMarkdown"]
