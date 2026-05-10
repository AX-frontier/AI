from __future__ import annotations

from dataclasses import dataclass

from agents.library.api.schemas import LibraryChatRequest, LibraryChatResponse
from agents.library.classifier import IntentClassification, classify_intent, extract_search_keyword
from agents.library.generator import (
    build_book_response,
    build_fallback_response,
    build_guide_response,
)
from agents.library.repository import LibraryRepository, get_library_repository
from agents.library.retrieval import (
    BookRetriever,
    GuideRetriever,
    BookSearchResult,
    GuideSearchResult,
    collect_retrieval_evidence,
)
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.llm.factory import get_llm_client


@dataclass(frozen=True)
class LibraryLLMPolicyDecision:
    use_answer_llm: bool
    use_intent_llm: bool
    reason: str


def run_library_agent(
    request: LibraryChatRequest,
    repository: LibraryRepository | None = None,
    llm_client: LLMClient | None = None,
) -> LibraryChatResponse:
    """Spring 요청을 받아 Library Agent 전체 파이프라인을 실행한다."""
    repo = repository or get_library_repository()
    book_retriever = BookRetriever(repo)
    guide_retriever = GuideRetriever(repo)

    message = request.message.strip()
    initial_classification = classify_intent(message)
    provisional_keyword = extract_search_keyword(message, initial_classification.intent)
    # 실제 검색 결과를 intent 판단에 반영하기 위해 두 저장소를 먼저 가볍게 조회한다.
    evidence = collect_retrieval_evidence(
        provisional_keyword,
        book_retriever=book_retriever,
        guide_retriever=guide_retriever,
        book_probe_limit=_book_probe_limit(initial_classification.intent),
        guide_probe_limit=_guide_probe_limit(initial_classification.intent),
        book_location_question=initial_classification.intent == "BOOK_LOCATION",
        include_probe_results=True,
    )
    classification = classify_intent(message, evidence=evidence)
    keyword = extract_search_keyword(message, classification.intent)

    if classification.intent in ("BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION"):
        location_question = classification.intent == "BOOK_LOCATION"
        books = _reuse_book_probe(
            evidence,
            keyword,
            required_limit=5,
            location_question=location_question,
        ) or book_retriever.retrieve(
            keyword,
            location_question=location_question,
        )
        return build_book_response(classification.intent, books, classification.confidence)

    guide_result = _reuse_guide_probe(evidence, keyword, required_limit=3) or guide_retriever.retrieve(
        keyword
    )
    llm_policy = decide_library_llm_policy(classification, guide_result)
    if guide_result.docs:
        context = guide_retriever.build_context(guide_result)
        return build_guide_response(
            classification.intent,
            context,
            classification.confidence,
            llm_client=(llm_client or _safe_get_llm_client())
            if llm_policy.use_answer_llm
            else None,
        )

    fallback_reason = (
        "관련 학술정보관 안내 문서를 찾지 못했습니다."
        if classification.intent == "LIBRARY_GUIDE"
        else "관련 안내 문서를 찾지 못했습니다."
    )
    return build_fallback_response(classification.intent, keyword, fallback_reason)


def decide_library_llm_policy(
    classification: IntentClassification,
    guide_result: GuideSearchResult | None = None,
) -> LibraryLLMPolicyDecision:
    """질문 분류와 검색 결과를 기준으로 LLM 호출 여부를 결정한다."""
    use_intent_llm = classification.ambiguous
    if classification.intent in ("BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION"):
        return LibraryLLMPolicyDecision(
            use_answer_llm=False,
            use_intent_llm=use_intent_llm,
            reason="book_intent_uses_structured_template",
        )

    if guide_result is None or not guide_result.docs:
        return LibraryLLMPolicyDecision(
            use_answer_llm=False,
            use_intent_llm=use_intent_llm,
            reason="no_guide_search_result",
        )

    if classification.intent == "LIBRARY_GUIDE" and (guide_result.chunks or guide_result.docs):
        return LibraryLLMPolicyDecision(
            use_answer_llm=True,
            use_intent_llm=use_intent_llm,
            reason="guide_search_result_available_for_rag_answer",
        )

    return LibraryLLMPolicyDecision(
        use_answer_llm=False,
        use_intent_llm=use_intent_llm,
        reason="answer_llm_not_required",
    )


def _safe_get_llm_client() -> LLMClient | None:
    try:
        return get_llm_client()
    except Exception:
        return None


def _reuse_book_probe(
    evidence,
    keyword: str,
    *,
    required_limit: int,
    location_question: bool,
) -> BookSearchResult | None:
    probe = evidence.book_probe
    if (
        probe is None
        or probe.keyword != keyword
        or evidence.book_probe_limit < required_limit
        or evidence.book_probe_location_question != location_question
    ):
        return None
    return probe


def _reuse_guide_probe(
    evidence,
    keyword: str,
    *,
    required_limit: int,
) -> GuideSearchResult | None:
    probe = evidence.guide_probe
    if (
        probe is None
        or probe.keyword != keyword
        or evidence.guide_probe_limit < required_limit
    ):
        return None
    return probe


def _book_probe_limit(intent: str) -> int:
    if intent in ("BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION"):
        return 5
    return 1


def _guide_probe_limit(intent: str) -> int:
    if intent in ("LIBRARY_GUIDE", "LIBRARY_GENERAL"):
        return 3
    return 1
