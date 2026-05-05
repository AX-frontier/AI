from __future__ import annotations

import re

from agents.library.api.schemas import LibraryChatResponse, LibraryIntent, LibrarySource, MatchedBook
from agents.library.models import BookRecord
from agents.library.retrieval import BookSearchResult, GuideContext


def build_book_response(
    intent: LibraryIntent,
    result: BookSearchResult,
    confidence: float,
) -> LibraryChatResponse:
    """도서 검색 결과를 Spring 호환 Library 응답으로 변환한다."""
    matched_books = [_book_to_response(book) for book in result.books]
    result_count = len(matched_books)

    if result_count == 0:
        return build_fallback_response(intent, result.keyword, "검색 조건과 일치하는 도서를 찾지 못했습니다.")

    first = matched_books[0]
    if intent == "BOOK_LOCATION":
        answer = (
            f"'{result.keyword}' 검색 결과 {result_count}건을 찾았습니다. "
            f"대표 도서 '{first.title}'의 청구기호는 {first.holdingCallNo or '정보 없음'}이고, "
            f"소장 위치는 {first.stackLocation or '정보 없음'}"
            f"{' ' + first.stackShelf if first.stackShelf else ''}입니다."
        )
    elif intent == "BOOK_RECOMMENDATION":
        titles = ", ".join(book.title for book in matched_books[:3])
        answer = f"'{result.keyword}'와 관련해 {result_count}건의 도서를 찾았습니다. 우선 {titles}을 확인해 보세요."
    else:
        titles = ", ".join(book.title for book in matched_books[:3])
        answer = f"'{result.keyword}' 검색 결과 {result_count}건을 찾았습니다. 주요 결과는 {titles}입니다."

    return LibraryChatResponse(
        intent=intent,
        answer=answer,
        sources=[],
        confidence=max(confidence, 0.8),
        fallbackUsed=False,
        fallbackReason=None,
        searchKeyword=result.keyword,
        resultCount=result_count,
        matchedBooks=matched_books,
    )


def build_guide_response(
    intent: LibraryIntent,
    context: GuideContext,
    confidence: float,
) -> LibraryChatResponse:
    """검색된 안내 컨텍스트와 출처 정보를 바탕으로 안내 QA 응답을 만든다."""
    if not context.docs:
        return build_fallback_response(intent, context.keyword, "관련 학술정보관 안내 문서를 찾지 못했습니다.")

    primary = context.docs[0]
    snippet = _summarize_content(primary.content)
    sources = [
        LibrarySource(
            id=doc.id,
            title=doc.title,
            sourceUrl=doc.source_url,
            updatedAt=doc.updated_at.isoformat() if doc.updated_at else None,
        )
        for doc in context.docs
    ]
    return LibraryChatResponse(
        intent=intent,
        answer=f"{primary.title}: {snippet}",
        sources=sources,
        confidence=max(confidence, 0.75),
        fallbackUsed=False,
        fallbackReason=None,
        searchKeyword=context.keyword,
        resultCount=0,
        matchedBooks=[],
    )


def build_fallback_response(intent: LibraryIntent, keyword: str, reason: str) -> LibraryChatResponse:
    """검색으로 답변하지 못했을 때 일관된 fallback 응답을 반환한다."""
    return LibraryChatResponse(
        intent=intent,
        answer=f"{reason} 다른 검색어로 다시 질문해 주세요.",
        sources=[],
        confidence=0.35,
        fallbackUsed=True,
        fallbackReason=reason,
        searchKeyword=keyword,
        resultCount=0,
        matchedBooks=[],
    )


def _book_to_response(book: BookRecord) -> MatchedBook:
    """DB의 snake_case 도서 필드를 Spring 응답용 camelCase 필드로 매핑한다."""
    return MatchedBook(
        id=book.id,
        bibNo=book.bib_no,
        regNo=book.reg_no,
        title=book.title,
        author=book.author,
        publisher=book.publisher,
        publishYear=book.publish_year,
        holdingCallNo=book.holding_call_no,
        materialType=book.material_type,
        locationSymbol=book.location_symbol,
        stackLocation=book.stack_location,
        stackShelf=book.stack_shelf,
    )


def _summarize_content(content: str, max_length: int = 280) -> str:
    """LLM 답변 생성기가 붙기 전까지 안내 답변을 짧게 요약한다."""
    compact = re.sub(r"\s+", " ", content).strip()
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "..."
