from __future__ import annotations

import html
import re

from agents.library.api.schemas import LibraryChatResponse, LibraryIntent, LibrarySource, MatchedBook
from agents.library.models import BookRecord
from agents.library.retrieval import BookSearchResult, GuideContext
from agents.main_agent.llm.base import LLMClient


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
    llm_client: LLMClient | None = None,
) -> LibraryChatResponse:
    """검색된 안내 컨텍스트와 출처 정보를 바탕으로 안내 QA 응답을 만든다."""
    if not context.docs:
        return build_fallback_response(intent, context.keyword, "관련 학술정보관 안내 문서를 찾지 못했습니다.")

    primary_chunk = context.chunks[0] if context.chunks else None
    primary = context.docs[0]
    answer_title = primary_chunk.title if primary_chunk else primary.title
    answer_content = primary_chunk.content if primary_chunk else primary.content
    answer = (
        _generate_guide_answer(context, llm_client)
        if llm_client
        else _compose_guide_answer(answer_title, answer_content)
    )
    sources = _build_guide_sources(context)
    return LibraryChatResponse(
        intent=intent,
        answer=answer,
        sources=sources,
        confidence=max(confidence, 0.75),
        fallbackUsed=False,
        fallbackReason=None,
        searchKeyword=context.keyword,
        resultCount=0,
        matchedBooks=[],
    )


def _compose_guide_answer(title: str, content: str) -> str:
    snippet = _summarize_content(content)
    return f"{title}: {snippet}"


def _generate_guide_answer(context: GuideContext, llm_client: LLMClient) -> str:
    import logging
    primary_chunk = context.chunks[0] if context.chunks else None
    primary = context.docs[0]
    fallback_title = primary_chunk.title if primary_chunk else primary.title
    fallback_content = primary_chunk.content if primary_chunk else primary.content
    try:
        answer = llm_client.generate(_build_guide_answer_prompt(context))
    except Exception as e:
        logging.getLogger(__name__).warning("LLM guide answer generation failed: %s", e)
        return _compose_guide_answer(fallback_title, fallback_content)
    return answer.strip() if answer and answer.strip() else _compose_guide_answer(
        fallback_title,
        fallback_content,
    )


def _build_guide_answer_prompt(context: GuideContext) -> str:
    context_blocks = []
    if context.chunks:
        for index, chunk in enumerate(context.chunks[:3], start=1):
            context_blocks.append(
                f"[{index}] title: {chunk.title}\n"
                f"source_url: {chunk.source_url or '출처 URL 없음'}\n"
                f"chunk_index: {chunk.chunk_index}\n"
                f"score: {chunk.score:.3f}\n"
                f"content:\n{_summarize_content(chunk.content, max_length=900)}"
            )
    else:
        for index, doc in enumerate(context.docs[:3], start=1):
            context_blocks.append(
                f"[{index}] title: {doc.title}\n"
                f"source_url: {doc.source_url or '출처 URL 없음'}\n"
                f"content:\n{_summarize_content(doc.content, max_length=900)}"
            )

    return (
        "당신은 한성대학교 학술정보관 안내를 담당하는 AI입니다.\n"
        "아래 검색 근거에 있는 내용만 사용해서 한국어로 간결하고 친절하게 답변하세요.\n"
        "근거에 없는 내용은 추측하지 말고 확인할 수 없다고 말하세요.\n"
        "답변 끝에는 참고한 출처 제목을 짧게 언급하세요.\n\n"
        f"사용자 질문: {context.keyword}\n\n"
        "검색 근거:\n"
        + "\n\n".join(context_blocks)
    )


def _build_guide_sources(context: GuideContext) -> list[LibrarySource]:
    sources: list[LibrarySource] = []
    seen: set[int] = set()
    for chunk in context.chunks:
        if chunk.guide_doc_id in seen:
            continue
        seen.add(chunk.guide_doc_id)
        sources.append(
            LibrarySource(
                id=chunk.guide_doc_id,
                title=chunk.title,
                sourceUrl=chunk.source_url,
                updatedAt=chunk.updated_at.isoformat() if chunk.updated_at else None,
            )
        )
    for doc in context.docs:
        if doc.id in seen:
            continue
        seen.add(doc.id)
        sources.append(
            LibrarySource(
                id=doc.id,
                title=doc.title,
                sourceUrl=doc.source_url,
                updatedAt=doc.updated_at.isoformat() if doc.updated_at else None,
            )
        )
    return sources


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
        bibNo=_decode_text(book.bib_no),
        regNo=_decode_text(book.reg_no),
        title=_decode_text(book.title) or "",
        author=_decode_text(book.author),
        publisher=_decode_text(book.publisher),
        publishYear=book.publish_year,
        holdingCallNo=_decode_text(book.holding_call_no),
        materialType=_decode_text(book.material_type),
        locationSymbol=_decode_text(book.location_symbol),
        stackLocation=_decode_text(book.stack_location),
        stackShelf=_decode_text(book.stack_shelf),
    )


def _decode_text(value: str | None) -> str | None:
    if value is None:
        return None
    return html.unescape(value)


def _summarize_content(content: str, max_length: int = 280) -> str:
    """LLM 답변 생성기가 붙기 전까지 안내 답변을 짧게 요약한다."""
    compact = re.sub(r"\s+", " ", content).strip()
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "..."
