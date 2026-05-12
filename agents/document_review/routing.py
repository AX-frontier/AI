from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentReviewEvidence:
    score: float
    reason: str


DOCUMENT_REVIEW_KEYWORDS = (
    "문서",
    "전자결재",
    "공문",
    "기안",
    "양식",
    "검토",
    "수정",
    "두문",
    "본문",
    "결문",
    "맞춤법",
    "문장",
)

# 시스템 사용법·절차를 묻는 맥락을 걸러낸다 (문서 검토 요청이 아님)
DOCUMENT_REVIEW_NEGATIVE_KEYWORDS = (
    "방법",
    "절차",
    "어떻게",
)

NEGATIVE_PENALTY = 0.20


def collect_document_review_evidence(message: str) -> DocumentReviewEvidence:
    """문서 검토 실행 없이 라우팅용 처리 가능성만 판단한다."""
    normalized = message.lower()
    matched = [keyword for keyword in DOCUMENT_REVIEW_KEYWORDS if keyword in normalized]
    if not matched:
        return DocumentReviewEvidence(score=0.0, reason="no document review keyword")

    score = min(0.95, 0.45 + (0.15 * len(matched)))
    if _is_explicit_document_review_request(normalized):
        score = max(score, 0.85)
    if _looks_like_review_command(normalized):
        score = max(score, 0.75)

    negative_matched = [kw for kw in DOCUMENT_REVIEW_NEGATIVE_KEYWORDS if kw in normalized]
    if negative_matched:
        score = max(0.0, score - NEGATIVE_PENALTY * len(negative_matched))

    reason = f"matched document review keywords: {', '.join(matched)}"
    if negative_matched:
        reason += f"; negative keywords: {', '.join(negative_matched)}"
    return DocumentReviewEvidence(score=round(score, 3), reason=reason)


def _looks_like_review_command(message: str) -> bool:
    return bool(re.search(r"(검토|수정|확인|고쳐|봐줘|봐 줘)", message))


def _is_explicit_document_review_request(message: str) -> bool:
    document_terms = ("문서", "공문", "전자결재", "기안", "양식")
    review_actions = ("검토", "수정", "확인", "고쳐", "봐줘", "봐 줘")
    if any(term in message for term in document_terms) and any(
        action in message for action in review_actions
    ):
        return True
    return "기안" in message and "문서" in message
