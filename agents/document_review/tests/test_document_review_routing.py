from __future__ import annotations

import pytest

from agents.document_review.routing import collect_document_review_evidence


@pytest.mark.parametrize(
    "message",
    [
        "이 공문 문장 검토해줘",
        "전자결재 문서를 수정해줘",
        "기안할 문서가 있는데 검토해줄 수 있어?",
        "문서 양식 수정해줘",
    ],
)
def test_document_review_evidence_scores_review_queries_high(message: str) -> None:
    evidence = collect_document_review_evidence(message)

    assert evidence.score >= 0.85
    assert "matched document review keywords" in evidence.reason


@pytest.mark.parametrize(
    "message",
    [
        "복수전공 신청 기간 알려줘",
        "도서관 운영시간 알려줘",
    ],
)
def test_document_review_evidence_scores_unrelated_queries_low(message: str) -> None:
    evidence = collect_document_review_evidence(message)

    assert evidence.score == 0.0
    assert evidence.reason == "no document review keyword"
