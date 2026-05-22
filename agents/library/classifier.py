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
GENERIC_LIBRARY_TERMS = ("도서관", "학술정보관", "학정관")
LOCATION_HINTS = ("위치", "어디", "소장", "서가", "자료실", "층", "청구기호")
BOOK_HINTS = ("검색", "찾아", "찾고", "도서", "책", "저자", "작가", "출판사", "관련")
BOOK_GENERIC_TERMS = ("책", "도서")
BOOK_TOPIC_HINTS = ("입문서", "교재", "전공서", "참고서")
BOOK_QUERY_SUFFIXES = ("있어", "있나요", "있니", "있는지", "보여줘", "알려줘")
BOOK_TOPIC_SUFFIX_REWRITES = (("입문서", "입문"),)
LEADING_POSTPOSITIONS = ("에", "에서", "의", "은", "는", "이", "가", "을", "를")
RECOMMENDATION_HINTS = (
    "추천",
    "볼만한",
    "읽을만",
    "읽을 만",
    "비슷한",
    "인기",
    "인기있는",
    "인기 있는",
    "베스트",
    "베스트셀러",
    "많이 읽",
    "유명",
)
POPULARITY_HINTS = (
    "인기",
    "인기있는",
    "인기 있는",
    "베스트",
    "베스트셀러",
    "많이 읽",
    "유명",
)
GUIDE_SPECIFIC_TERMS = (
    "규정",
    "세칙",
    "조직",
    "조직안내",
    "직원",
    "담당",
    "담당자",
    "누구",
    "연락처",
    "메일",
    "이메일",
    "전화",
    "전화번호",
    "장서관리",
    "자료수서",
    "회원제",
    "faq",
    "운영",
    "시간",
    "이용",
    "휴관",
    "열람실",
    "좌석",
    "예약",
    "사물함",
    "시설",
    "반납",
    "대출",
    "연장",
    "연체료",
    "납부",
    "결제",
    "채납",
    "수납",
    "회원",
    "문의",
    "전자책",
    "전자자료",
    "db",
    "개관",
)
GUIDE_PAYMENT_TERMS = ("연체", "연체료", "납부", "결제", "채납", "수납", "벌금", "미납")
GUIDE_POLICY_TERMS = ("규정", "세칙", "조직", "조직안내", "회원제", "faq")
GUIDE_STAFF_TERMS = (
    "관장",
    "직원",
    "담당",
    "담당자",
    "누구",
    "연락처",
    "메일",
    "이메일",
    "전화",
    "전화번호",
    "장서관리",
    "자료수서",
)
GUIDE_TIME_QUESTION_HINTS = ("오늘", "몇 시", "까지", "열어", "열어요", "닫", "오픈")
GENERAL_HINTS = (
    "안녕",
    "안녕하세요",
    "도와줘",
    "궁금",
    "문의하고 싶",
    "뭘 물어봐야",
    "모르겠어",
)
GENERAL_INQUIRY_PHRASES = ("문의하고 싶은", "문의가 있어")
BOOK_FIELD_QUALIFIERS = ("저자", "작가", "출판사", "청구기호", "서가", "자료")


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
    book_probe_limit: int = 0
    guide_probe_limit: int = 0
    book_probe_location_question: bool = False
    book_probe: Any | None = None
    guide_probe: Any | None = None


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
    has_generic_book_term = _contains_any(cleaned, BOOK_GENERIC_TERMS)

    for phrase in sorted(intent_config.keyword_cleanup, key=len, reverse=True):
        cleaned = cleaned.replace(phrase, " ")

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:;")

    if intent in ("BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION"):
        original_cleaned = cleaned
        if "말고" in cleaned:
            cleaned = cleaned.rsplit("말고", 1)[-1].strip()
        suffix_pattern = "|".join(re.escape(term) for term in BOOK_QUERY_SUFFIXES)
        cleaned = re.sub(rf"\s*(?:{suffix_pattern})\s*$", "", cleaned).strip()
        cleaned = re.sub(r"\s*(책|도서)[이가을를은는]?\s*$", "", cleaned).strip()
        cleaned = re.sub(r"(학술\s*정보관|도서관)", " ", cleaned).strip()
        qualifier_pattern = "|".join(re.escape(term) for term in BOOK_FIELD_QUALIFIERS)
        cleaned = re.sub(
            rf"(?:\s+(?:{qualifier_pattern})[이가을를은는]?\s*)+$",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        leading_particle_pattern = "|".join(re.escape(term) for term in LEADING_POSTPOSITIONS)
        cleaned = re.sub(rf"^(?:{leading_particle_pattern})\s+", "", cleaned).strip()
        for source, replacement in BOOK_TOPIC_SUFFIX_REWRITES:
            cleaned = re.sub(rf"{re.escape(source)}$", replacement, cleaned).strip()
        if not cleaned and has_generic_book_term:
            return "책"

    return cleaned or message.strip()


def _score_rule(rule: IntentRule, normalized: str, evidence: RetrievalEvidence) -> IntentScore:
    """현재 메시지에 대해 intent 규칙 하나를 숫자 점수로 변환한다."""
    matched = _matched_keywords(rule.keywords, normalized)
    negative = tuple(keyword for keyword in rule.negative_keywords if keyword.lower() in normalized)
    raw_score = float(len(matched)) - (1.2 * len(negative))

    evidence_bonus = _evidence_bonus(rule.intent, evidence, normalized)
    heuristic_bonus = _heuristic_bonus(rule.intent, normalized, evidence)
    priority_bonus = (
        min(rule.priority / 100.0, 0.6)
        if matched or evidence_bonus > 0 or heuristic_bonus > 0
        else 0.0
    )
    adjusted = raw_score + evidence_bonus + heuristic_bonus + priority_bonus

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
            "heuristicBonus": round(heuristic_bonus, 3),
        },
    )


def _evidence_bonus(intent: LibraryIntent, evidence: RetrievalEvidence, normalized: str) -> float:
    """해당 저장소에 실제 검색 결과가 있으면 관련 intent 점수를 올린다."""
    if intent in ("BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION") and evidence.book_hits:
        return 1.4
    if intent == "LIBRARY_GUIDE" and evidence.guide_hits:
        return (
            1.4
            if _contains_any(normalized, GUIDE_SPECIFIC_TERMS + GUIDE_TIME_QUESTION_HINTS)
            else 0.45
        )
    if intent == "LIBRARY_GENERAL" and evidence.guide_hits:
        return 0.4
    return 0.0


def _heuristic_bonus(intent: LibraryIntent, normalized: str, evidence: RetrievalEvidence) -> float:
    """규칙 사전만으로 놓치기 쉬운 경계 질문을 보정한다."""
    bonus = 0.0
    has_location_hint = _has_location_hint(normalized)
    has_book_hint = _contains_any(normalized, BOOK_HINTS)
    has_recommendation_hint = _contains_any(normalized, RECOMMENDATION_HINTS)
    has_popular_book_hint = _contains_any(normalized, POPULARITY_HINTS) and _contains_any(
        normalized,
        BOOK_GENERIC_TERMS,
    )
    has_generic_library_term = _contains_any(normalized, GENERIC_LIBRARY_TERMS)
    has_guide_specific_term = _contains_any(normalized, GUIDE_SPECIFIC_TERMS)
    has_guide_payment_term = _contains_any(normalized, GUIDE_PAYMENT_TERMS)
    has_guide_policy_term = _contains_any(normalized, GUIDE_POLICY_TERMS)
    has_guide_staff_term = _contains_any(normalized, GUIDE_STAFF_TERMS)
    has_guide_time_question = _contains_any(normalized, GUIDE_TIME_QUESTION_HINTS)
    looks_general = _looks_like_general_message(normalized)
    prefers_physical_book_search = (
        "말고" in normalized and "전자책" in normalized and ("종이책" in normalized or "책" in normalized)
    )
    rejects_recommendation = "추천 말고" in normalized

    if intent == "BOOK_SEARCH":
        if has_popular_book_hint:
            bonus -= 1.5
        if has_generic_library_term and has_guide_policy_term and not has_book_hint:
            bonus -= 1.2
        if _contains_any(normalized, BOOK_TOPIC_HINTS):
            bonus += 1.2
        if has_generic_library_term and has_book_hint:
            bonus += 0.9
        if has_location_hint:
            bonus -= 0.85
        if has_recommendation_hint and "말고" not in normalized:
            bonus -= 0.7
        if rejects_recommendation:
            bonus += 1.0
        if prefers_physical_book_search:
            bonus += 1.8
        if has_guide_specific_term and not has_book_hint:
            bonus -= 0.8
        if has_guide_staff_term and not has_book_hint:
            bonus -= 1.4

    if intent == "BOOK_LOCATION":
        if has_popular_book_hint:
            bonus -= 0.9
        if has_generic_library_term and has_guide_policy_term and not has_book_hint:
            bonus -= 1.2
        if has_guide_payment_term and not has_book_hint:
            bonus -= 1.4
        if has_guide_staff_term and not has_book_hint:
            bonus -= 1.3
        if has_location_hint:
            bonus += 1.05
        if has_location_hint and has_book_hint:
            bonus += 0.55
        if _contains_any(normalized, ("서가", "자료실", "청구기호")):
            bonus += 0.35
        if evidence.book_hits and has_location_hint:
            bonus += 0.2

    if intent == "BOOK_RECOMMENDATION":
        if has_popular_book_hint:
            bonus += 1.75
        if has_generic_library_term and has_guide_policy_term and not has_recommendation_hint:
            bonus -= 1.2
        if has_recommendation_hint and has_book_hint and not rejects_recommendation:
            bonus += 1.45
        if has_recommendation_hint and _contains_any(normalized, ("가장", "제일", "top", "탑")):
            bonus += 0.55
        if has_location_hint:
            bonus -= 0.65
        if rejects_recommendation and has_recommendation_hint:
            bonus -= 2.5
        if has_guide_staff_term and not has_book_hint:
            bonus -= 1.2

    if intent == "LIBRARY_GUIDE":
        if has_generic_library_term and has_guide_policy_term:
            bonus += 1.4
        if has_guide_specific_term:
            bonus += 0.8
        if has_guide_payment_term:
            bonus += 1.25
        if has_guide_staff_term:
            bonus += 1.45
        if has_guide_time_question:
            bonus += 0.7
        if prefers_physical_book_search:
            bonus -= 1.2
        if _contains_any(normalized, GENERAL_INQUIRY_PHRASES) and not has_location_hint:
            bonus -= 1.15
        if has_generic_library_term and has_book_hint:
            bonus -= 1.0
        if looks_general:
            bonus -= 1.1

    if intent == "LIBRARY_GENERAL" and looks_general:
        bonus += 1.6

    return bonus


def _contains_any(normalized: str, terms: tuple[str, ...]) -> bool:
    compact_normalized = _compact_text(normalized)
    for term in terms:
        normalized_term = term.lower()
        if normalized_term in normalized:
            return True
        if _compact_text(normalized_term) in compact_normalized:
            return True
    return False


def _matched_keywords(keywords: tuple[str, ...], normalized: str) -> tuple[str, ...]:
    compact_normalized = _compact_text(normalized)
    matched = []
    for keyword in keywords:
        normalized_keyword = keyword.lower()
        if normalized_keyword not in normalized and _compact_text(normalized_keyword) not in compact_normalized:
            continue
        if normalized_keyword == "서가" and not _is_valid_shelf_keyword(normalized):
            continue
        matched.append(keyword)
    return tuple(matched)


def _has_location_hint(normalized: str) -> bool:
    terms_without_shelf = tuple(term for term in LOCATION_HINTS if term != "서가")
    return _contains_any(normalized, terms_without_shelf) or _is_valid_shelf_keyword(normalized)


def _is_valid_shelf_keyword(normalized: str) -> bool:
    """문서가/문서가요 내부의 '서가' 오탐을 피하고 실제 서가 문맥만 인정한다."""
    if not re.search(r"(?<!문)서가", normalized):
        return False
    shelf_context_terms = (
        "책",
        "도서",
        "자료실",
        "청구기호",
        "위치",
        "어디",
        "소장",
        "층",
        "도서관",
        "학술정보관",
    )
    return _contains_any(normalized, shelf_context_terms) or bool(
        re.search(r"\b\d+[ -]?[a-z]-\d+", normalized)
    )


def _looks_like_general_message(normalized: str) -> bool:
    if _contains_any(normalized, GENERAL_INQUIRY_PHRASES):
        return not (_contains_any(normalized, BOOK_HINTS) or _has_location_hint(normalized))
    if not _contains_any(normalized, GENERAL_HINTS):
        return False
    return not (
        _contains_any(normalized, BOOK_HINTS + GUIDE_SPECIFIC_TERMS)
        or _has_location_hint(normalized)
    )


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


def _compact_text(text: str) -> str:
    return "".join(text.split())
