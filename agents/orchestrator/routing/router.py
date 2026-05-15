from __future__ import annotations

from dataclasses import dataclass

from agents.orchestrator.api.schemas import TargetAgent
from agents.orchestrator.routing.evidence import RoutingEvidence

DOCUMENT_REVIEW_THRESHOLD = 0.70
LIBRARY_THRESHOLD = 0.70


@dataclass(frozen=True)
class RouteDecision:
    target_agent: TargetAgent
    intent: str
    confidence: float
    reason: str


class EvidenceBasedRouter:
    """정책 기반으로 targetAgent를 결정한다.

    정책:
    - DOCUMENT_REVIEW는 문서검토 증거가 임계치 이상일 때만 선택
    - LIBRARY는 도서검색 전용 증거가 임계치 이상일 때만 선택
    - 그 외 모든 질의는 MAIN으로 라우팅
    """

    def route(self, evidence: RoutingEvidence) -> RouteDecision:
        if evidence.document_review.score >= 0.85:
            return _decision_for(
                "DOCUMENT_REVIEW",
                evidence,
                "explicit document review evidence selected",
            )

        if evidence.document_review.score >= DOCUMENT_REVIEW_THRESHOLD:
            return _decision_for(
                "DOCUMENT_REVIEW",
                evidence,
                "document review evidence exceeded threshold",
            )

        if evidence.library.score >= LIBRARY_THRESHOLD:
            return _decision_for(
                "LIBRARY",
                evidence,
                "book-search library evidence exceeded threshold",
            )

        return _decision_for(
            "MAIN",
            evidence,
            "default route: non-library and non-document-review request",
        )


def _decision_for(target: str, evidence: RoutingEvidence, prefix: str) -> RouteDecision:
    if target == "DOCUMENT_REVIEW":
        return RouteDecision(
            target_agent="DOCUMENT_REVIEW",
            intent="DOCUMENT_REVIEW",
            confidence=evidence.document_review.score,
            reason=f"{prefix}: {evidence.document_review.reason}",
        )
    if target == "LIBRARY":
        return RouteDecision(
            target_agent="LIBRARY",
            intent="LIBRARY",
            confidence=evidence.library.score,
            reason=f"{prefix}: {evidence.library.reason}",
        )
    if target == "MAIN":
        return RouteDecision(
            target_agent="MAIN",
            intent="MAIN",
            confidence=evidence.main.score,
            reason=f"{prefix}: {evidence.main.reason}",
        )
    return RouteDecision(
        target_agent="FALLBACK",
        intent="FALLBACK",
        confidence=0.0,
        reason="fallback selected",
    )
