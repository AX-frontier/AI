from __future__ import annotations

from dataclasses import dataclass, field

from agents.orchestrator.api.schemas import TargetAgent
from agents.orchestrator.routing.evidence import RoutingEvidence

DOCUMENT_REVIEW_THRESHOLD = 0.70
LIBRARY_THRESHOLD = 0.65
RULE_ADJUST_LIMIT = 0.12
MAIN_NOTICE_TERMS = ("공지", "한성공지", "최신", "새로 올라온")
LIBRARY_TERMS = ("도서관", "학술정보관", "대출", "반납", "연장", "열람실", "운영", "개관", "휴관", "책", "도서")
DOC_TERMS = ("문서", "전자결재", "검토", "교정", "맞춤법", "기안", "공문")


@dataclass(frozen=True)
class RouteDecision:
    target_agent: TargetAgent
    intent: str
    confidence: float
    reason: str
    raw_scores: dict[str, float] = field(default_factory=dict)
    rule_adjustments: dict[str, float] = field(default_factory=dict)
    final_scores: dict[str, float] = field(default_factory=dict)


class EvidenceBasedRouter:
    """정책 기반으로 targetAgent를 결정한다.

    정책:
    - DOCUMENT_REVIEW는 문서검토 증거가 임계치 이상일 때만 선택
    - LIBRARY는 도서검색 전용 증거가 임계치 이상일 때만 선택
    - 그 외 모든 질의는 MAIN으로 라우팅
    """

    def route(self, evidence: RoutingEvidence, message: str = "") -> RouteDecision:
        if evidence.document_review.score >= 0.85:
            return _decision_for(
                "DOCUMENT_REVIEW",
                evidence,
                "explicit document review evidence selected",
                raw_scores=_raw_scores(evidence),
                rule_adjustments={"MAIN": 0.0, "LIBRARY": 0.0, "DOCUMENT_REVIEW": 0.0},
                final_scores=_raw_scores(evidence),
            )
        if evidence.document_review.score >= DOCUMENT_REVIEW_THRESHOLD:
            raw_scores = _raw_scores(evidence)
            return _decision_for(
                "DOCUMENT_REVIEW",
                evidence,
                _compose_reason(
                    "document review evidence exceeded threshold",
                    raw_scores,
                    {"MAIN": 0.0, "LIBRARY": 0.0, "DOCUMENT_REVIEW": 0.0},
                    raw_scores,
                ),
                raw_scores=raw_scores,
                rule_adjustments={"MAIN": 0.0, "LIBRARY": 0.0, "DOCUMENT_REVIEW": 0.0},
                final_scores=raw_scores,
            )

        raw_scores = _raw_scores(evidence)
        rule_adjustments = _rule_adjustments(message)
        final_scores = {
            agent: _clamp(raw_scores[agent] + rule_adjustments.get(agent, 0.0))
            for agent in raw_scores
        }
        ranked = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        top_agent, top_score = ranked[0]
        second_score = ranked[1][1]
        if message.strip() and top_agent == "LIBRARY" and not _has_library_domain_terms(message):
            return _decision_for(
                "MAIN",
                evidence,
                _compose_reason("library domain-mismatch fallback to MAIN", raw_scores, rule_adjustments, final_scores),
                raw_scores=raw_scores,
                rule_adjustments=rule_adjustments,
                final_scores=final_scores,
            )

        if top_agent == "DOCUMENT_REVIEW" and top_score >= DOCUMENT_REVIEW_THRESHOLD:
            return _decision_for(
                "DOCUMENT_REVIEW",
                evidence,
                _compose_reason("document review final score selected", raw_scores, rule_adjustments, final_scores),
                raw_scores=raw_scores,
                rule_adjustments=rule_adjustments,
                final_scores=final_scores,
            )
        if top_agent == "LIBRARY" and top_score >= LIBRARY_THRESHOLD:
            return _decision_for(
                "LIBRARY",
                evidence,
                _compose_reason("library final score selected", raw_scores, rule_adjustments, final_scores),
                raw_scores=raw_scores,
                rule_adjustments=rule_adjustments,
                final_scores=final_scores,
            )
        if top_agent == "MAIN":
            return _decision_for(
                "MAIN",
                evidence,
                _compose_reason("main final score selected", raw_scores, rule_adjustments, final_scores),
                raw_scores=raw_scores,
                rule_adjustments=rule_adjustments,
                final_scores=final_scores,
            )
        if top_score - second_score < 0.02:
            return _decision_for(
                "MAIN",
                evidence,
                _compose_reason("tie-break favored MAIN", raw_scores, rule_adjustments, final_scores),
                raw_scores=raw_scores,
                rule_adjustments=rule_adjustments,
                final_scores=final_scores,
            )
        return _decision_for(
            "MAIN",
            evidence,
            _compose_reason("default route after score synthesis", raw_scores, rule_adjustments, final_scores),
            raw_scores=raw_scores,
            rule_adjustments=rule_adjustments,
            final_scores=final_scores,
        )


def _decision_for(
    target: str,
    evidence: RoutingEvidence,
    prefix: str,
    *,
    raw_scores: dict[str, float] | None = None,
    rule_adjustments: dict[str, float] | None = None,
    final_scores: dict[str, float] | None = None,
) -> RouteDecision:
    if target == "DOCUMENT_REVIEW":
        return RouteDecision(
            target_agent="DOCUMENT_REVIEW",
            intent="DOCUMENT_REVIEW",
            confidence=evidence.document_review.score,
            reason=f"{prefix}: {evidence.document_review.reason}",
            raw_scores=raw_scores or {},
            rule_adjustments=rule_adjustments or {},
            final_scores=final_scores or {},
        )
    if target == "LIBRARY":
        return RouteDecision(
            target_agent="LIBRARY",
            intent="LIBRARY",
            confidence=evidence.library.score,
            reason=f"{prefix}: {evidence.library.reason}",
            raw_scores=raw_scores or {},
            rule_adjustments=rule_adjustments or {},
            final_scores=final_scores or {},
        )
    if target == "MAIN":
        return RouteDecision(
            target_agent="MAIN",
            intent="MAIN",
            confidence=evidence.main.score,
            reason=f"{prefix}: {evidence.main.reason}",
            raw_scores=raw_scores or {},
            rule_adjustments=rule_adjustments or {},
            final_scores=final_scores or {},
        )
    return RouteDecision(
        target_agent="FALLBACK",
        intent="FALLBACK",
        confidence=0.0,
        reason="fallback selected",
        raw_scores=raw_scores or {},
        rule_adjustments=rule_adjustments or {},
        final_scores=final_scores or {},
    )


def _raw_scores(evidence: RoutingEvidence) -> dict[str, float]:
    return {
        "MAIN": _clamp(evidence.main.score),
        "LIBRARY": _clamp(evidence.library.score),
        "DOCUMENT_REVIEW": _clamp(evidence.document_review.score),
    }


def _rule_adjustments(message: str) -> dict[str, float]:
    lowered = (message or "").lower()
    main_adj = 0.0
    library_adj = 0.0
    doc_adj = 0.0
    if any(term in lowered for term in MAIN_NOTICE_TERMS):
        main_adj += 0.10
        library_adj -= 0.05
    if any(term in lowered for term in LIBRARY_TERMS):
        library_adj += 0.10
    if any(term in lowered for term in DOC_TERMS):
        doc_adj += 0.10
        main_adj -= 0.03
    return {
        "MAIN": _cap_adjust(main_adj),
        "LIBRARY": _cap_adjust(library_adj),
        "DOCUMENT_REVIEW": _cap_adjust(doc_adj),
    }


def _cap_adjust(value: float) -> float:
    return max(-RULE_ADJUST_LIMIT, min(RULE_ADJUST_LIMIT, value))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _compose_reason(
    prefix: str,
    raw_scores: dict[str, float],
    rule_adjustments: dict[str, float],
    final_scores: dict[str, float],
) -> str:
    return (
        f"{prefix}; "
        f"raw={{main:{raw_scores['MAIN']:.3f},library:{raw_scores['LIBRARY']:.3f},doc:{raw_scores['DOCUMENT_REVIEW']:.3f}}} "
        f"rule={{main:{rule_adjustments['MAIN']:+.2f},library:{rule_adjustments['LIBRARY']:+.2f},doc:{rule_adjustments['DOCUMENT_REVIEW']:+.2f}}} "
        f"final={{main:{final_scores['MAIN']:.3f},library:{final_scores['LIBRARY']:.3f},doc:{final_scores['DOCUMENT_REVIEW']:.3f}}}"
    )


def _has_library_domain_terms(message: str) -> bool:
    lowered = (message or "").lower()
    return any(term in lowered for term in LIBRARY_TERMS)
