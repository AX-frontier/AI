from __future__ import annotations

from dataclasses import dataclass

from agents.orchestrator.api.schemas import TargetAgent
from agents.orchestrator.routing.evidence import RoutingEvidence

DOCUMENT_REVIEW_THRESHOLD = 0.70
LIBRARY_THRESHOLD = 0.70
MAIN_VECTOR_THRESHOLD = 0.45
AMBIGUOUS_MARGIN = 0.10


@dataclass(frozen=True)
class RouteDecision:
    target_agent: TargetAgent
    intent: str
    confidence: float
    reason: str


class EvidenceBasedRouter:
    """Evidence 점수와 안전 우선순위로 targetAgent를 결정한다."""

    def route(self, evidence: RoutingEvidence) -> RouteDecision:
        candidates = [
            ("DOCUMENT_REVIEW", evidence.document_review.score, DOCUMENT_REVIEW_THRESHOLD),
            ("LIBRARY", evidence.library.score, LIBRARY_THRESHOLD),
            ("MAIN", evidence.main.score, MAIN_VECTOR_THRESHOLD),
        ]
        eligible = [
            (target, score, threshold)
            for target, score, threshold in candidates
            if score >= threshold
        ]
        if not eligible:
            return RouteDecision(
                target_agent="FALLBACK",
                intent="FALLBACK",
                confidence=0.0,
                reason="no evidence exceeded routing thresholds",
            )

        best_target, best_score, _ = max(eligible, key=lambda item: item[1])
        second_score = sorted((score for _, score, _ in eligible), reverse=True)[1:2]
        if second_score and best_score - second_score[0] < AMBIGUOUS_MARGIN:
            priority_target = _priority_pick(eligible)
            return _decision_for(priority_target, evidence, "ambiguous scores; priority policy used")

        return _decision_for(best_target, evidence, "highest evidence score selected")


def _priority_pick(eligible: list[tuple[str, float, float]]) -> str:
    targets = {target for target, _, _ in eligible}
    for target in ("DOCUMENT_REVIEW", "LIBRARY", "MAIN"):
        if target in targets:
            return target
    return "FALLBACK"


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
