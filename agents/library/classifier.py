from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import yaml

from agents.library.api.schemas import LibraryIntent

CONFIG_PATH = Path(__file__).parent / "config" / "intents.yaml"
AMBIGUOUS_MARGIN = 0.35
LOW_CONFIDENCE_THRESHOLD = 0.55


@dataclass(frozen=True)
class IntentRule:
    """YAML에 정의된 intent별 점수 계산 규칙."""

    intent: LibraryIntent
    priority: int
    min_score: float
    keywords: tuple[str, ...]
    negative_keywords: tuple[str, ...]
    examples: tuple[str, ...]


@dataclass(frozen=True)
class IntentConfig:
    """로드된 intent 사전과 검색어에서 제거할 문구 목록."""

    rules: tuple[IntentRule, ...]
    keyword_cleanup: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalEvidence:
    """intent 판단을 보강하기 위한 가벼운 DB 검색 결과 수."""

    book_hits: int = 0
    guide_hits: int = 0
    book_keyword: str | None = None
    guide_keyword: str | None = None


@dataclass(frozen=True)
class IntentScore:
    """최종 intent를 고르기 전 후보 intent 하나의 점수."""

    intent: LibraryIntent
    raw_score: float
    confidence: float
    matched_keywords: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntentClassification:
    """Library Agent 파이프라인에서 사용하는 최종 라우팅 결과."""

    intent: LibraryIntent
    confidence: float
    reason: str
    evidence: dict[str, Any]
    ambiguous: bool = False
    used_llm: bool = False


class LLMIntentClassifier(Protocol):
    """애매한 질문에서만 사용할 수 있는 선택형 LLM 분류기 인터페이스."""

    def classify(
        self,
        message: str,
        candidates: list[IntentScore],
        evidence: RetrievalEvidence,
    ) -> IntentClassification | None:
        ...


class DisabledLLMIntentClassifier:
    """로컬/개발 환경에서 API 키 없이 동작하도록 하는 no-op LLM 분류기."""

    def classify(
        self,
        message: str,
        candidates: list[IntentScore],
        evidence: RetrievalEvidence,
    ) -> IntentClassification | None:
        return None


@lru_cache(maxsize=1)
def load_intent_config(config_path: str | None = None) -> IntentConfig:
    """YAML intent 규칙을 한 번 로드하고 요청마다 재사용한다."""
    path = Path(config_path) if config_path else CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    rules = []
    for intent_name, rule_payload in (payload.get("intents") or {}).items():
        rules.append(
            IntentRule(
                intent=intent_name,
                priority=int(rule_payload.get("priority", 0)),
                min_score=float(rule_payload.get("minScore", 0)),
                keywords=tuple(rule_payload.get("keywords") or ()),
                negative_keywords=tuple(rule_payload.get("negativeKeywords") or ()),
                examples=tuple(rule_payload.get("examples") or ()),
            )
        )
    rules.sort(key=lambda rule: rule.priority, reverse=True)
    return IntentConfig(
        rules=tuple(rules),
        keyword_cleanup=tuple(payload.get("keywordCleanup") or ()),
    )


def classify_intent(
    message: str,
    evidence: RetrievalEvidence | None = None,
    llm_classifier: LLMIntentClassifier | None = None,
    config: IntentConfig | None = None,
) -> IntentClassification:
    """YAML 점수, 검색 evidence, 선택형 LLM 순서로 최적 intent를 고른다."""
    intent_config = config or load_intent_config()
    retrieval_evidence = evidence or RetrievalEvidence()
    scores = score_intents(message, retrieval_evidence, intent_config)
    best = scores[0]
    second = scores[1] if len(scores) > 1 else None
    ambiguous = _is_ambiguous(best, second)

    if ambiguous:
        llm_result = (llm_classifier or _default_llm_classifier()).classify(
            message,
            scores[:3],
            retrieval_evidence,
        )
        if llm_result is not None:
            return llm_result

    reason = _build_reason(best, retrieval_evidence, ambiguous)
    return IntentClassification(
        intent=best.intent,
        confidence=best.confidence,
        reason=reason,
        evidence=best.evidence,
        ambiguous=ambiguous,
        used_llm=False,
    )


def score_intents(
    message: str,
    evidence: RetrievalEvidence,
    config: IntentConfig | None = None,
) -> list[IntentScore]:
    """설정된 모든 intent에 점수를 매기고 강한 후보부터 반환한다."""
    intent_config = config or load_intent_config()
    normalized = message.lower()
    scores = [_score_rule(rule, normalized, evidence) for rule in intent_config.rules]
    scores.sort(key=lambda score: (score.confidence, score.raw_score), reverse=True)
    return scores


def extract_search_keyword(
    message: str,
    intent: LibraryIntent | None = None,
    config: IntentConfig | None = None,
) -> str:
    """저장소 검색에 필요한 핵심 키워드만 남기기 위해 명령형 문구를 제거한다."""
    intent_config = config or load_intent_config()
    cleaned = message.strip()
    cleaned = re.sub(r"[?？!！.。]+$", "", cleaned).strip()

    for phrase in sorted(intent_config.keyword_cleanup, key=len, reverse=True):
        cleaned = cleaned.replace(phrase, " ")

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:;")
    return cleaned or message.strip()


def _score_rule(rule: IntentRule, normalized: str, evidence: RetrievalEvidence) -> IntentScore:
    """현재 메시지에 대해 intent 규칙 하나를 숫자 점수로 변환한다."""
    matched = tuple(keyword for keyword in rule.keywords if keyword.lower() in normalized)
    negative = tuple(keyword for keyword in rule.negative_keywords if keyword.lower() in normalized)
    raw_score = float(len(matched)) - (1.2 * len(negative))

    evidence_bonus = _evidence_bonus(rule.intent, evidence)
    priority_bonus = min(rule.priority / 100.0, 0.6)
    adjusted = raw_score + evidence_bonus + priority_bonus

    if rule.intent == "LIBRARY_GENERAL" and adjusted <= 0:
        confidence = 0.4
    elif adjusted < rule.min_score:
        confidence = max(0.05, adjusted / max(rule.min_score, 1.0) * 0.45)
    else:
        confidence = min(0.98, 0.5 + (adjusted / 5.0))

    return IntentScore(
        intent=rule.intent,
        raw_score=adjusted,
        confidence=round(confidence, 3),
        matched_keywords=matched,
        evidence={
            "matchedKeywords": list(matched),
            "negativeKeywords": list(negative),
            "bookHits": evidence.book_hits,
            "guideHits": evidence.guide_hits,
        },
    )


def _evidence_bonus(intent: LibraryIntent, evidence: RetrievalEvidence) -> float:
    """해당 저장소에 실제 검색 결과가 있으면 관련 intent 점수를 올린다."""
    if intent in ("BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION") and evidence.book_hits:
        return 1.4
    if intent in ("LIBRARY_GUIDE", "LIBRARY_GENERAL") and evidence.guide_hits:
        return 1.4
    return 0.0


def _is_ambiguous(best: IntentScore, second: IntentScore | None) -> bool:
    """확신도가 낮거나 점수 차이가 작아 LLM 보조가 필요한 상황을 감지한다."""
    if best.intent == "LIBRARY_GENERAL":
        return best.confidence < LOW_CONFIDENCE_THRESHOLD
    if best.confidence < LOW_CONFIDENCE_THRESHOLD:
        return True
    if second is None:
        return False
    return best.confidence - second.confidence < AMBIGUOUS_MARGIN


def _build_reason(best: IntentScore, evidence: RetrievalEvidence, ambiguous: bool) -> str:
    """로그나 오케스트레이터 메타데이터에 쓸 짧은 판단 근거를 만든다."""
    parts = []
    if best.matched_keywords:
        parts.append(f"matched keywords: {', '.join(best.matched_keywords)}")
    if evidence.book_hits:
        parts.append(f"book search hits: {evidence.book_hits}")
    if evidence.guide_hits:
        parts.append(f"guide search hits: {evidence.guide_hits}")
    if ambiguous:
        parts.append("low margin; deterministic fallback used")
    return "; ".join(parts) or "no strong intent signal; defaulted deterministically"


def _default_llm_classifier() -> LLMIntentClassifier:
    """설정된 LLM 분류기를 반환한다. 현재는 의도적으로 비활성화되어 있다."""
    if os.getenv("LIBRARY_LLM_CLASSIFIER_ENABLED", "false").lower() != "true":
        return DisabledLLMIntentClassifier()
    return DisabledLLMIntentClassifier()
