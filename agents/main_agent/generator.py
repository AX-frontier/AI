from __future__ import annotations

import re

from agents.main_agent.api.schemas import MainChatResponse, MainSource
from agents.main_agent.models import MainChunkRecord
from agents.main_agent.retrieval import MainSearchResult


def build_main_response(result: MainSearchResult) -> MainChatResponse:
    """검색된 chunk를 Spring 호환 Main Agent 응답으로 변환한다."""
    if not result.chunks:
        return build_main_fallback_response(result.keyword, "관련 학교 공지 또는 안내 chunk를 찾지 못했습니다.")

    primary = result.chunks[0]
    answer = _compose_answer(primary)
    sources = [_chunk_to_source(chunk) for chunk in result.chunks]
    confidence = max(0.5, min(0.95, primary.score))
    return MainChatResponse(
        intent=_infer_main_intent(primary),
        answer=answer,
        sources=sources,
        confidence=round(confidence, 3),
        fallbackUsed=False,
        fallbackReason=None,
        searchKeyword=result.keyword,
        resultCount=len(result.chunks),
    )


def build_main_fallback_response(keyword: str, reason: str) -> MainChatResponse:
    return MainChatResponse(
        intent="MAIN_GENERAL",
        answer=f"{reason} 공식 공지나 학사 안내 데이터가 적재된 뒤 다시 확인해 주세요.",
        sources=[],
        confidence=0.35,
        fallbackUsed=True,
        fallbackReason=reason,
        searchKeyword=keyword,
        resultCount=0,
    )


def _compose_answer(chunk: MainChunkRecord) -> str:
    snippet = _summarize_text(chunk.text)
    posted = f" 게시일은 {chunk.posted_date}입니다." if chunk.posted_date else ""
    return f"{chunk.title}: {snippet}{posted}"


def _chunk_to_source(chunk: MainChunkRecord) -> MainSource:
    return MainSource(
        title=chunk.title,
        url=chunk.url,
        documentId=chunk.document_id,
        chunkId=chunk.chunk_id,
        category=chunk.category,
        postedDate=chunk.posted_date,
        score=round(chunk.score, 3),
    )


def _infer_main_intent(chunk: MainChunkRecord) -> str:
    category = (chunk.category or "").lower()
    if "학사" in category or "수강" in chunk.text or "전공" in chunk.text:
        return "ACADEMIC_INFO_QA"
    if "공지" in category or chunk.url:
        return "SCHOOL_NOTICE_QA"
    return "MAIN_GENERAL"


def _summarize_text(text: str, max_length: int = 320) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "..."
