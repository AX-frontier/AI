from __future__ import annotations

from agents.library.api.schemas import LibraryChatRequest, LibraryChatResponse
from agents.library.classifier import classify_intent, extract_search_keyword
from agents.library.generator import (
    build_book_response,
    build_fallback_response,
    build_guide_response,
)
from agents.library.repository import LibraryRepository, get_library_repository
from agents.library.retrieval import BookRetriever, GuideRetriever, collect_retrieval_evidence


def run_library_agent(
    request: LibraryChatRequest,
    repository: LibraryRepository | None = None,
) -> LibraryChatResponse:
    repo = repository or get_library_repository()
    book_retriever = BookRetriever(repo)
    guide_retriever = GuideRetriever(repo)

    message = request.message.strip()
    provisional_keyword = extract_search_keyword(message)
    evidence = collect_retrieval_evidence(
        provisional_keyword,
        book_retriever=book_retriever,
        guide_retriever=guide_retriever,
    )
    classification = classify_intent(message, evidence=evidence)
    keyword = extract_search_keyword(message, classification.intent)

    if classification.intent in ("BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION"):
        books = book_retriever.retrieve(keyword)
        return build_book_response(classification.intent, books, classification.confidence)

    guide_result = guide_retriever.retrieve(keyword)
    if guide_result.docs:
        context = guide_retriever.build_context(guide_result)
        return build_guide_response(classification.intent, context, classification.confidence)

    fallback_reason = (
        "관련 학술정보관 안내 문서를 찾지 못했습니다."
        if classification.intent == "LIBRARY_GUIDE"
        else "관련 안내 문서를 찾지 못했습니다."
    )
    return build_fallback_response(classification.intent, keyword, fallback_reason)

