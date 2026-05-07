from __future__ import annotations

import re

from agents.main_agent.api.schemas import MainChatResponse, MainSource
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.models import MainChunkRecord
from agents.main_agent.retrieval import MainSearchResult

MAIN_VECTOR_THRESHOLD = 0.45


def build_main_response(
    result: MainSearchResult,
    llm_client: LLMClient | None = None,
) -> MainChatResponse:
    """검색된 chunk를 Spring 호환 Main Agent 응답으로 변환한다."""
    if not result.chunks:
        return build_main_fallback_response(result.keyword, "관련 학교 공지 또는 안내 chunk를 찾지 못했습니다.")

    primary = result.chunks[0]
    if primary.score < MAIN_VECTOR_THRESHOLD:
        return build_main_fallback_response(
            result.keyword,
            "관련 학교 공지 또는 안내 chunk의 유사도가 낮습니다.",
        )

    answer = _generate_answer(result, llm_client) if llm_client else _compose_answer(primary)
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


def _generate_answer(result: MainSearchResult, llm_client: LLMClient | None) -> str:
    if llm_client is None:
        return _compose_answer(result.chunks[0])

    try:
        answer = llm_client.generate(_build_answer_prompt(result))
    except Exception:
        return _compose_answer(result.chunks[0])

    return answer or _compose_answer(result.chunks[0])


def _build_answer_prompt(result: MainSearchResult) -> str:
    context_blocks = []
    for index, chunk in enumerate(result.chunks[:5], start=1):
        source = chunk.url or "출처 URL 없음"
        context_blocks.append(
            f"[{index}] title: {chunk.title}\n"
            f"url: {source}\n"
            f"posted_date: {chunk.posted_date or '정보 없음'}\n"
            f"content:\n{_summarize_text(chunk.text, max_length=900)}"
        )

    return (
        "당신은 한성대학교 공식 공지와 학사 안내를 바탕으로 답변하는 AI입니다.\n"
        "아래 검색 근거에 있는 내용만 사용해서 한국어로 간결하게 답변하세요.\n"
        "근거에 없는 내용은 추측하지 말고 확인할 수 없다고 말하세요.\n"
        "답변 끝에는 참고한 출처 제목을 짧게 언급하세요.\n\n"
        f"사용자 질문: {result.keyword}\n\n"
        "검색 근거:\n"
        + "\n\n".join(context_blocks)
    )


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
