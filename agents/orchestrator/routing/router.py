from __future__ import annotations

from dataclasses import dataclass, field

from agents.orchestrator.api.schemas import TargetAgent
from agents.orchestrator.routing.evidence import RoutingEvidence

DOCUMENT_REVIEW_THRESHOLD = 0.70
LIBRARY_THRESHOLD = 0.65
CAMPUS_MAP_THRESHOLD = 0.62
RULE_ADJUST_LIMIT = 0.12
MAIN_NOTICE_TERMS = ("공지", "한성공지", "최신", "새로 올라온")
LIBRARY_TERMS = ("도서관", "학술정보관", "대출", "반납", "연장", "열람실", "운영", "개관", "휴관", "책", "도서")
DOC_TERMS = ("문서", "전자결재", "검토", "교정", "맞춤법", "기안", "공문")
CAMPUS_LOCATION_TERMS = ("어디", "위치", "가는 길", "가는길", "길찾기", "출입구", "정문", "후문")
CAMPUS_PLACE_TERMS = ("상상관", "학생회관", "학술정보관", "도서관", "공학관", "미래관", "탐구관", "진리관", "창의관", "우촌관", "인성관", "낙산관")
CAMPUS_BUILDING_TERMS = ("상상관", "학생회관", "공학관", "미래관", "탐구관", "진리관", "창의관", "우촌관", "인성관", "낙산관")
BOOK_LOCATION_TERMS = ("책", "도서", "청구기호", "서가", "소장")


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
                rule_adjustments={"MAIN": 0.0, "LIBRARY": 0.0, "DOCUMENT_REVIEW": 0.0, "CAMPUS_MAP": 0.0},
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
                    {"MAIN": 0.0, "LIBRARY": 0.0, "DOCUMENT_REVIEW": 0.0, "CAMPUS_MAP": 0.0},
                    raw_scores,
                ),
                raw_scores=raw_scores,
                rule_adjustments={"MAIN": 0.0, "LIBRARY": 0.0, "DOCUMENT_REVIEW": 0.0, "CAMPUS_MAP": 0.0},
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
        if top_agent == "CAMPUS_MAP" and _has_book_location_terms(message):
            return _decision_for(
                "LIBRARY" if final_scores.get("LIBRARY", 0.0) >= 0.4 else "MAIN",
                evidence,
                _compose_reason("book location query avoided CAMPUS_MAP", raw_scores, rule_adjustments, final_scores),
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
        if top_agent == "CAMPUS_MAP" and top_score >= CAMPUS_MAP_THRESHOLD:
            return _decision_for(
                "CAMPUS_MAP",
                evidence,
                _compose_reason("campus map final score selected", raw_scores, rule_adjustments, final_scores),
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
    if target == "CAMPUS_MAP":
        return RouteDecision(
            target_agent="CAMPUS_MAP",
            intent="CAMPUS_MAP",
            confidence=evidence.campus_map.score,
            reason=f"{prefix}: {evidence.campus_map.reason}",
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
        "CAMPUS_MAP": _clamp(evidence.campus_map.score),
    }


def _rule_adjustments(message: str) -> dict[str, float]:
    lowered = (message or "").lower()
    main_adj = 0.0
    library_adj = 0.0
    doc_adj = 0.0
    campus_adj = 0.0
    if _contains_any_term(lowered, MAIN_NOTICE_TERMS):
        main_adj += 0.10
        library_adj -= 0.05
    if _contains_any_term(lowered, LIBRARY_TERMS):
        library_adj += 0.10
    if _contains_any_term(lowered, DOC_TERMS):
        doc_adj += 0.10
        main_adj -= 0.03
    has_campus_location_intent = _contains_any_term(lowered, CAMPUS_LOCATION_TERMS) and _contains_any_term(lowered, CAMPUS_PLACE_TERMS)
    has_non_library_building = _contains_any_term(lowered, CAMPUS_BUILDING_TERMS)
    if has_campus_location_intent or has_non_library_building:
        campus_adj += 0.12
        if not _has_book_location_terms(lowered):
            library_adj -= 0.08
    return {
        "MAIN": _cap_adjust(main_adj),
        "LIBRARY": _cap_adjust(library_adj),
        "DOCUMENT_REVIEW": _cap_adjust(doc_adj),
        "CAMPUS_MAP": _cap_adjust(campus_adj),
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
        f"raw={{main:{raw_scores['MAIN']:.3f},library:{raw_scores['LIBRARY']:.3f},doc:{raw_scores['DOCUMENT_REVIEW']:.3f},campus:{raw_scores['CAMPUS_MAP']:.3f}}} "
        f"rule={{main:{rule_adjustments['MAIN']:+.2f},library:{rule_adjustments['LIBRARY']:+.2f},doc:{rule_adjustments['DOCUMENT_REVIEW']:+.2f},campus:{rule_adjustments['CAMPUS_MAP']:+.2f}}} "
        f"final={{main:{final_scores['MAIN']:.3f},library:{final_scores['LIBRARY']:.3f},doc:{final_scores['DOCUMENT_REVIEW']:.3f},campus:{final_scores['CAMPUS_MAP']:.3f}}}"
    )


def _has_library_domain_terms(message: str) -> bool:
    lowered = (message or "").lower()
    return _contains_any_term(lowered, LIBRARY_TERMS)


def _has_book_location_terms(message: str) -> bool:
    lowered = (message or "").lower()
    return _contains_any_term(lowered, BOOK_LOCATION_TERMS) and not _contains_any_term(lowered, ("도서관", "학술정보관"))


def _contains_any_term(message: str, terms: tuple[str, ...]) -> bool:
    compact_message = _compact_text(message)
    for term in terms:
        normalized_term = term.lower()
        if normalized_term in message:
            return True
        if _compact_text(normalized_term) in compact_message:
            return True
    return False


def _compact_text(text: str) -> str:
    return "".join(text.split())
