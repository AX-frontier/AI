from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from agents.library.aladin import AladinClient
from agents.library.api.schemas import LibraryChatRequest, LibraryChatResponse
from agents.library.classifier import IntentClassification, classify_intent, extract_search_keyword
from agents.library.generator import (
    build_book_response,
    build_clarification_response,
    build_fallback_options_response,
    build_guide_response,
)
from agents.library.repository import LibraryRepository, get_library_repository
from agents.library.recommendation import build_recommendation_summary, rank_book_recommendations
from agents.library.recommendation_query import RecommendationQuery, interpret_recommendation_query
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


_FOLLOWUP_MEMORY: dict[str, dict[str, str]] = {}


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
    recommendation_query = (
        interpret_recommendation_query(message, keyword, llm_client=llm_client)
        if classification.intent == "BOOK_RECOMMENDATION"
        else None
    )

    if classification.intent in ("BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION"):
        if classification.intent == "BOOK_RECOMMENDATION" and _is_generic_book_recommendation_message(message, keyword):
            response = _build_book_recommendation_clarification(keyword)
            _remember_context(str(request.conversationUid), classification.intent, keyword)
            return response
        if _should_clarify_message(message, keyword, classification.intent, evidence):
            response = _build_ambiguous_clarification(classification.intent, keyword)
            _remember_context(str(request.conversationUid), classification.intent, keyword)
            return response
        location_question = classification.intent == "BOOK_LOCATION"
        retrieval_keyword = _recommendation_retrieval_keyword(keyword, recommendation_query)
        if classification.intent == "BOOK_RECOMMENDATION" and recommendation_query and recommendation_query.rewritten:
            books = _retrieve_recommendation_books(
                recommendation_query,
                keyword,
                book_retriever,
                location_question=location_question,
            )
        else:
            books = _reuse_book_probe(
                evidence,
                keyword,
                required_limit=5,
                location_question=location_question,
            ) if recommendation_query is None else None
            books = books or book_retriever.retrieve(
                retrieval_keyword,
                location_question=location_question,
                include_semantic=classification.intent == "BOOK_RECOMMENDATION",
            )
            if classification.intent == "BOOK_RECOMMENDATION" and not books.books:
                books = _retrieve_recommendation_books(
                    recommendation_query,
                    keyword,
                    book_retriever,
                    location_question=location_question,
                )
        if not books.books:
            books = _retrieve_expanded_books(
                retrieval_keyword,
                book_retriever,
                location_question=location_question,
            )
        if classification.intent == "BOOK_RECOMMENDATION" and books.books:
            ranking_keyword = _recommendation_ranking_keyword(books.keyword, recommendation_query)
            ranked = rank_book_recommendations(
                ranking_keyword,
                books.books,
                aladin_weight=_recommendation_aladin_weight(recommendation_query),
                expected_kdc=_recommendation_expected_kdc(recommendation_query),
                recommendation_mode=_recommendation_mode(recommendation_query),
            )
            books = BookSearchResult(books.keyword, [item.book for item in ranked])
            summary = build_recommendation_summary(
                ranked,
                expandedFrom=keyword if recommendation_query and recommendation_query.rewritten else (
                    keyword if books.keyword != keyword else None
                ),
                includeSemantic=_recommendation_uses_semantic(recommendation_query),
                queryInterpretation=_query_interpretation_summary(recommendation_query),
                aladinReference=_aladin_reference_summary(recommendation_query),
            )
            response = build_book_response(
                classification.intent,
                books,
                classification.confidence,
                summary=summary,
            )
        elif books.books:
            response = build_book_response(classification.intent, books, classification.confidence)
        else:
            response = _build_book_fallback_options(classification.intent, keyword)
        _remember_context(str(request.conversationUid), classification.intent, keyword)
        return response

    guide_result = _reuse_guide_probe(evidence, keyword, required_limit=3) or guide_retriever.retrieve(
        keyword
    )
    if not guide_result.docs:
        guide_result = _retrieve_expanded_guides(keyword, guide_retriever)
    llm_policy = decide_library_llm_policy(classification, guide_result)
    if guide_result.docs:
        context = guide_retriever.build_context(guide_result)
        response = build_guide_response(
            classification.intent,
            context,
            classification.confidence,
            llm_client=(llm_client or _safe_get_llm_client())
            if llm_policy.use_answer_llm
            else None,
        )
        topic = None
        if response.summary and isinstance(response.summary, dict):
            title = response.summary.get("title")
            if isinstance(title, str) and title.strip():
                topic = title.strip()
        _remember_context(str(request.conversationUid), classification.intent, keyword, topic=topic)
        return response

    fallback_reason = (
        "관련 학술정보관 안내 문서를 찾지 못했습니다."
        if classification.intent == "LIBRARY_GUIDE"
        else "관련 안내 문서를 찾지 못했습니다."
    )
    response = _build_guide_fallback_options(classification.intent, keyword, fallback_reason)
    _remember_context(str(request.conversationUid), classification.intent, keyword)
    return response


def run_library_agent_stream(
    request: LibraryChatRequest,
    repository: LibraryRepository | None = None,
    llm_client: LLMClient | None = None,
) -> Iterator[dict]:
    """Library 에이전트를 streaming 모드로 실행한다.

    Yields:
        {'type': 'chunk', 'text': str} — LLM 응답 토큰 또는 완성된 답변
        {'type': 'done', ...} — 최종 응답 필드 (model_dump() 전개)
    """
    from agents.library.generator import (
        build_book_response,
        build_guide_response,
        build_guide_response_with_answer,
        _build_guide_answer_prompt,
        _compose_guide_answer,
    )

    repo = repository or get_library_repository()
    book_retriever = BookRetriever(repo)
    guide_retriever = GuideRetriever(repo)

    message = request.message.strip()
    initial_classification = classify_intent(message)
    provisional_keyword = extract_search_keyword(message, initial_classification.intent)
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
    recommendation_query = (
        interpret_recommendation_query(message, keyword, llm_client=llm_client)
        if classification.intent == "BOOK_RECOMMENDATION"
        else None
    )

    if classification.intent in ("BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION"):
        if classification.intent == "BOOK_RECOMMENDATION" and _is_generic_book_recommendation_message(message, keyword):
            response = _build_book_recommendation_clarification(keyword)
            _remember_context(str(request.conversationUid), classification.intent, keyword)
            yield {"type": "done", **response.model_dump()}
            return
        if _should_clarify_message(message, keyword, classification.intent, evidence):
            response = _build_ambiguous_clarification(classification.intent, keyword)
            _remember_context(str(request.conversationUid), classification.intent, keyword)
            yield {"type": "done", **response.model_dump()}
            return
        location_question = classification.intent == "BOOK_LOCATION"
        retrieval_keyword = _recommendation_retrieval_keyword(keyword, recommendation_query)
        if classification.intent == "BOOK_RECOMMENDATION" and recommendation_query and recommendation_query.rewritten:
            books = _retrieve_recommendation_books(
                recommendation_query,
                keyword,
                book_retriever,
                location_question=location_question,
            )
        else:
            books = _reuse_book_probe(
                evidence, keyword, required_limit=5, location_question=location_question
            ) if recommendation_query is None else None
            books = books or book_retriever.retrieve(
                retrieval_keyword,
                location_question=location_question,
                include_semantic=classification.intent == "BOOK_RECOMMENDATION",
            )
            if classification.intent == "BOOK_RECOMMENDATION" and not books.books:
                books = _retrieve_recommendation_books(
                    recommendation_query,
                    keyword,
                    book_retriever,
                    location_question=location_question,
                )
        if not books.books:
            books = _retrieve_expanded_books(
                retrieval_keyword,
                book_retriever,
                location_question=location_question,
            )
        if classification.intent == "BOOK_RECOMMENDATION" and books.books:
            ranking_keyword = _recommendation_ranking_keyword(books.keyword, recommendation_query)
            ranked = rank_book_recommendations(
                ranking_keyword,
                books.books,
                aladin_weight=_recommendation_aladin_weight(recommendation_query),
                expected_kdc=_recommendation_expected_kdc(recommendation_query),
                recommendation_mode=_recommendation_mode(recommendation_query),
            )
            books = BookSearchResult(books.keyword, [item.book for item in ranked])
            summary = build_recommendation_summary(
                ranked,
                expandedFrom=keyword if recommendation_query and recommendation_query.rewritten else (
                    keyword if books.keyword != keyword else None
                ),
                includeSemantic=_recommendation_uses_semantic(recommendation_query),
                queryInterpretation=_query_interpretation_summary(recommendation_query),
                aladinReference=_aladin_reference_summary(recommendation_query),
            )
            response = build_book_response(
                classification.intent,
                books,
                classification.confidence,
                summary=summary,
            )
        elif books.books:
            response = build_book_response(classification.intent, books, classification.confidence)
        else:
            response = _build_book_fallback_options(classification.intent, keyword)
        _remember_context(str(request.conversationUid), classification.intent, keyword)
        yield {"type": "done", **response.model_dump()}
        return

    guide_result = _reuse_guide_probe(evidence, keyword, required_limit=3) or guide_retriever.retrieve(keyword)
    if not guide_result.docs:
        guide_result = _retrieve_expanded_guides(keyword, guide_retriever)
    llm_policy = decide_library_llm_policy(classification, guide_result)

    if guide_result.docs:
        context = guide_retriever.build_context(guide_result)
        if llm_policy.use_answer_llm and llm_client:
            prompt = _build_guide_answer_prompt(context)
            accumulated = ""
            try:
                for chunk_text in llm_client.generate_stream(prompt):
                    accumulated += chunk_text
                    yield {"type": "chunk", "text": chunk_text}
            except Exception:
                primary_chunk = context.chunks[0] if context.chunks else None
                primary = context.docs[0]
                fallback_title = primary_chunk.title if primary_chunk else primary.title
                fallback_content = primary_chunk.content if primary_chunk else primary.content
                accumulated = _compose_guide_answer(fallback_title, fallback_content, context.keyword)
                yield {"type": "chunk", "text": accumulated}
            response = build_guide_response_with_answer(
                classification.intent, context, classification.confidence, accumulated
            )
        else:
            response = build_guide_response(classification.intent, context, classification.confidence, llm_client=None)
        yield {"type": "done", **response.model_dump()}
        topic = None
        if response.summary and isinstance(response.summary, dict):
            title = response.summary.get("title")
            if isinstance(title, str) and title.strip():
                topic = title.strip()
        _remember_context(str(request.conversationUid), classification.intent, keyword, topic=topic)
        return

    fallback_reason = (
        "관련 학술정보관 안내 문서를 찾지 못했습니다."
        if classification.intent == "LIBRARY_GUIDE"
        else "관련 안내 문서를 찾지 못했습니다."
    )
    response = _build_guide_fallback_options(classification.intent, keyword, fallback_reason)
    _remember_context(str(request.conversationUid), classification.intent, keyword)
    yield {"type": "done", **response.model_dump()}


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


def _should_clarify_message(message: str, keyword: str, intent: str, evidence) -> bool:
    normalized = message.strip().lower()
    if evidence.book_hits or evidence.guide_hits:
        return False
    if intent == "BOOK_RECOMMENDATION" and not _is_generic_recommendation_keyword(keyword):
        return False
    if intent == "BOOK_LOCATION":
        return False
    ambiguous_phrases = (
        "자료 찾아줘",
        "자료 검색",
        "추천해줘",
        "어디서 봐",
        "어디서 보면",
        "어디에서 봐",
    )
    if any(phrase in normalized for phrase in ambiguous_phrases):
        return True
    return keyword.strip() in {"자료", "추천해줘", "추천", "어디서 봐"}


def _is_generic_book_recommendation_message(message: str, keyword: str) -> bool:
    normalized_message = message.strip().lower()
    if "책" not in normalized_message and "도서" not in normalized_message:
        return False
    return _is_generic_recommendation_keyword(keyword)


def _is_generic_recommendation_keyword(keyword: str) -> bool:
    normalized = keyword.strip().lower()
    return normalized in {"", "책", "도서", "추천", "추천해줘"}


def _build_ambiguous_clarification(intent: str, keyword: str):
    return build_clarification_response(
        intent,
        keyword,
        "도서 추천/검색을 원하시나요, 학술정보관 이용 안내를 원하시나요?",
        [
            {"value": "book", "label": "도서 추천/검색"},
            {"value": "guide", "label": "학술정보관 이용 안내"},
        ],
    )


def _build_book_recommendation_clarification(keyword: str):
    return build_clarification_response(
        "BOOK_RECOMMENDATION",
        keyword,
        "어떤 기준으로 도서를 추천해 드릴까요?",
        [
            {"value": "popular", "label": "인기 도서"},
            {"value": "major", "label": "전공/분야별"},
            {"value": "light", "label": "가볍게 읽을 책"},
            {"value": "recent_intro", "label": "최근 출간/입문서"},
        ],
    )


def _build_book_fallback_options(intent: str, keyword: str):
    return build_fallback_options_response(
        intent,
        keyword,
        "소장 도서에서는 검색 조건과 일치하는 도서를 찾지 못했습니다.",
        [
            {"value": "similar_books", "label": "비슷한 주제 도서 추천"},
            {"value": "aladin_popular", "label": "알라딘 인기 도서 확인"},
            {"value": "library_guide", "label": "학술정보관 이용 안내 보기"},
        ],
    )


def _build_guide_fallback_options(intent: str, keyword: str, reason: str):
    return build_fallback_options_response(
        intent,
        keyword,
        reason,
        [
            {"value": "hours", "label": "운영시간"},
            {"value": "loan_return", "label": "대출/반납"},
            {"value": "seats_facilities", "label": "좌석/시설"},
            {"value": "electronic_resources", "label": "전자자료"},
        ],
    )


def _recommendation_retrieval_keyword(
    keyword: str,
    recommendation_query: RecommendationQuery | None,
) -> str:
    if recommendation_query and recommendation_query.search_queries:
        return recommendation_query.search_queries[0]
    if recommendation_query and recommendation_query.rewritten:
        return recommendation_query.display_keyword
    return keyword


def _recommendation_ranking_keyword(
    keyword: str,
    recommendation_query: RecommendationQuery | None,
) -> str:
    if recommendation_query and recommendation_query.rewritten and recommendation_query.search_queries:
        if recommendation_query.mode == "light_reading":
            return recommendation_query.display_keyword
        return " ".join(recommendation_query.search_queries)
    if recommendation_query and recommendation_query.rewritten:
        return recommendation_query.display_keyword
    return keyword


def _retrieve_recommendation_books(
    recommendation_query: RecommendationQuery | None,
    keyword: str,
    retriever: BookRetriever,
    *,
    location_question: bool,
) -> BookSearchResult:
    queries = recommendation_query.search_queries if recommendation_query else (keyword,)
    per_query_results: list[tuple[str, list]] = []
    used_queries = []
    aladin_books = _retrieve_aladin_referenced_books(recommendation_query, retriever)
    if aladin_books:
        per_query_results.append(("알라딘 베스트셀러", aladin_books))
        used_queries.append("알라딘 베스트셀러")
    for query in queries:
        result = retriever.retrieve(
            query,
            limit=3,
            location_question=location_question,
            include_semantic=True,
        )
        if result.books:
            per_query_results.append((query, result.books))
            used_queries.append(query)
    collected = _round_robin_books(per_query_results, limit=12)
    if collected:
        return BookSearchResult(used_queries[0] if used_queries else keyword, collected)
    return BookSearchResult(keyword, [])


def _retrieve_aladin_referenced_books(
    recommendation_query: RecommendationQuery | None,
    retriever: BookRetriever,
) -> list:
    if recommendation_query is None or recommendation_query.mode != "popular":
        return []
    try:
        signals = AladinClient().bestsellers(max_results=20)
    except Exception:
        return []

    collected = []
    seen_ids: set[int] = set()
    for signal in signals:
        for title_query in _aladin_title_queries(signal.title):
            result = retriever.retrieve(title_query, limit=2, include_semantic=False)
            matched = False
            for book in result.books:
                if book.id in seen_ids:
                    continue
                seen_ids.add(book.id)
                collected.append(book)
                matched = True
                break
            if matched:
                break
    return collected[:6]


def _aladin_title_queries(title: str) -> tuple[str, ...]:
    normalized = " ".join(title.replace("\xa0", " ").split())
    candidates = [normalized]
    for separator in (" - ", " : ", " (", " ["):
        if separator in normalized:
            candidates.append(normalized.split(separator, 1)[0].strip())
    seen: set[str] = set()
    return tuple(candidate for candidate in candidates if candidate and not (candidate in seen or seen.add(candidate)))


def _round_robin_books(per_query_results: list[tuple[str, list]], *, limit: int):
    collected = []
    seen_ids: set[int] = set()
    max_len = max((len(books) for _, books in per_query_results), default=0)
    for index in range(max_len):
        for _query, books in per_query_results:
            if index >= len(books):
                continue
            book = books[index]
            if book.id in seen_ids:
                continue
            seen_ids.add(book.id)
            collected.append(book)
            if len(collected) >= limit:
                return collected
    return collected


def _query_interpretation_summary(
    recommendation_query: RecommendationQuery | None,
) -> dict | None:
    if recommendation_query is None or not recommendation_query.rewritten:
        return None
    return {
        "rawKeyword": recommendation_query.raw_keyword,
        "displayKeyword": recommendation_query.display_keyword,
        "searchQueries": list(recommendation_query.search_queries),
        "mode": recommendation_query.mode,
        "expectedKdc": list(recommendation_query.expected_kdc),
    }


def _aladin_reference_summary(
    recommendation_query: RecommendationQuery | None,
) -> dict | None:
    if recommendation_query is None or recommendation_query.mode != "popular":
        return None
    return {
        "source": "aladin_bestseller",
        "matchingPolicy": "알라딘 베스트셀러 목록을 조회한 뒤 ISBN/제목으로 library.books 소장 도서와 대조합니다.",
        "recommendationScope": "library_holdings_only",
    }


def _recommendation_aladin_weight(
    recommendation_query: RecommendationQuery | None,
) -> float:
    if recommendation_query is None:
        return 1.0
    if recommendation_query.mode == "popular":
        return 1.8
    if recommendation_query.mode == "light_reading":
        return 1.35
    return 1.0


def _recommendation_expected_kdc(
    recommendation_query: RecommendationQuery | None,
) -> tuple[str, ...]:
    if recommendation_query is None:
        return ()
    return recommendation_query.expected_kdc


def _recommendation_mode(
    recommendation_query: RecommendationQuery | None,
) -> str | None:
    if recommendation_query is None:
        return None
    return recommendation_query.mode


def _recommendation_uses_semantic(
    recommendation_query: RecommendationQuery | None,
) -> bool:
    if recommendation_query is None:
        return True
    return bool(recommendation_query.search_queries)


def _retrieve_expanded_books(
    keyword: str,
    retriever: BookRetriever,
    *,
    location_question: bool,
) -> BookSearchResult:
    for expanded_keyword in _expanded_book_keywords(keyword):
        if expanded_keyword == keyword:
            continue
        result = retriever.retrieve(
            expanded_keyword,
            location_question=location_question,
        )
        if result.books:
            return result
    return BookSearchResult(keyword, [])


def _retrieve_expanded_guides(keyword: str, retriever: GuideRetriever) -> GuideSearchResult:
    for expanded_keyword in _expanded_guide_keywords(keyword):
        if expanded_keyword == keyword:
            continue
        result = retriever.retrieve(expanded_keyword)
        if result.docs:
            return result
    return GuideSearchResult(keyword, [], [])


def _expanded_book_keywords(keyword: str) -> list[str]:
    normalized = keyword.strip()
    expansions = {
        "딥러닝": ["머신러닝", "인공지능", "AI"],
        "인공지능": ["AI", "머신러닝", "딥러닝"],
        "ai": ["인공지능", "머신러닝", "딥러닝"],
        "프로그래밍": ["파이썬", "자바", "컴퓨터"],
        "코딩": ["프로그래밍", "파이썬"],
        "데이터": ["데이터베이스", "통계", "분석"],
        "경제": ["경영", "회계", "금융"],
        "문학": ["소설", "시", "에세이"],
    }
    candidates = [normalized]
    lowered = normalized.lower()
    for trigger, values in expansions.items():
        if trigger in lowered:
            candidates.extend(values)
    seen: set[str] = set()
    return [candidate for candidate in candidates if candidate and not (candidate in seen or seen.add(candidate))]


def _expanded_guide_keywords(keyword: str) -> list[str]:
    normalized = keyword.strip()
    lowered = normalized.lower()
    candidates = [normalized]
    if any(term in lowered for term in ("운영", "시간", "언제", "열어", "개관", "휴관")):
        candidates.extend(["운영 시간", "개관시간", "개관 시간", "휴관일"])
    if any(term in lowered for term in ("대출", "반납", "연장", "연체")):
        candidates.extend(["대출", "반납", "대출 반납", "연장", "연체"])
    if any(term in lowered for term in ("좌석", "열람실", "시설", "사물함", "예약")):
        candidates.extend(["좌석", "열람실", "시설", "예약"])
    if any(term in lowered for term in ("전자", "db", "데이터베이스")):
        candidates.extend(["전자자료", "전자책", "DB", "학술DB"])
    seen: set[str] = set()
    return [candidate for candidate in candidates if candidate and not (candidate in seen or seen.add(candidate))]


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


def _apply_followup_context(conversation_uid: str, message: str) -> str:
    if not _looks_followup(message):
        return message
    memory = _FOLLOWUP_MEMORY.get(conversation_uid)
    if not memory:
        return message
    base_topic = memory.get("topic") or memory.get("keyword")
    if not base_topic:
        return message
    return f"{base_topic} {message}"


def _remember_context(
    conversation_uid: str,
    intent: str,
    keyword: str,
    topic: str | None = None,
) -> None:
    _FOLLOWUP_MEMORY[conversation_uid] = {
        "intent": intent,
        "keyword": keyword,
        "topic": topic or keyword,
    }
    if len(_FOLLOWUP_MEMORY) > 500:
        oldest = next(iter(_FOLLOWUP_MEMORY.keys()))
        _FOLLOWUP_MEMORY.pop(oldest, None)


def _looks_followup(message: str) -> bool:
    lowered = message.strip().lower()
    if len(lowered) > 40:
        return False
    return any(
        token in lowered
        for token in (
            "그럼",
            "그러면",
            "그거",
            "그건",
            "거긴",
            "거기",
            "토요일은",
            "일요일은",
            "오늘은",
            "지금은",
        )
    )
