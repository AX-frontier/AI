from __future__ import annotations

import re
from dataclasses import dataclass

from agents.main_agent.api.schemas import MainChatResponse, MainSource
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.models import MainChunkRecord
from agents.main_agent.retrieval import MainSearchResult

MAIN_VECTOR_THRESHOLD = 0.45


@dataclass(frozen=True)
class _SourceLink:
    title: str
    url: str
    category: str | None
    posted_date: str | None
    snippet: str


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

    answer = _generate_answer(result, llm_client) if llm_client else _compose_answer(result)
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


def _compose_answer(result: MainSearchResult) -> str:
    links = _top_source_links(result.chunks)
    descriptions = [_default_link_description(link) for link in links]
    return _format_link_guide_answer(result.keyword, links, descriptions)


def _generate_answer(result: MainSearchResult, llm_client: LLMClient | None) -> str:
    if llm_client is None:
        return _compose_answer(result)

    links = _top_source_links(result.chunks)
    if not links:
        return _format_link_guide_answer(result.keyword, links, [])

    try:
        raw_answer = llm_client.generate(_build_link_description_prompt(result.keyword, links))
    except Exception:
        descriptions = [_default_link_description(link) for link in links]
        return _format_link_guide_answer(result.keyword, links, descriptions)

    descriptions = _parse_link_descriptions(raw_answer, len(links))
    if descriptions is None:
        descriptions = [_default_link_description(link) for link in links]
    return _format_link_guide_answer(result.keyword, links, descriptions)


def _top_source_links(chunks: list[MainChunkRecord], limit: int = 3) -> list[_SourceLink]:
    links: list[_SourceLink] = []
    seen_urls: set[str] = set()
    for chunk in chunks:
        if not chunk.url or chunk.url in seen_urls:
            continue
        links.append(
            _SourceLink(
                title=chunk.title,
                url=chunk.url,
                category=chunk.category,
                posted_date=chunk.posted_date,
                snippet=_summarize_text(chunk.text),
            )
        )
        seen_urls.add(chunk.url)
        if len(links) >= limit:
            break
    return links


def _build_link_description_prompt(keyword: str, links: list[_SourceLink]) -> str:
    source_blocks = []
    for index, link in enumerate(links, start=1):
        source_blocks.append(
            f"[{index}]\n"
            f"title: {link.title}\n"
            f"url: {link.url}\n"
            f"category: {link.category or '정보 없음'}\n"
            f"posted_date: {link.posted_date or '정보 없음'}\n"
            f"snippet: {link.snippet}"
        )
    return (
        "당신은 한성대학교 공식 공지와 학사 안내 검색 결과를 설명하는 AI입니다.\n"
        "아래 source별로 사용자가 왜 이 링크를 확인하면 좋은지 한국어 1문장으로만 설명하세요.\n"
        "URL은 절대 출력하지 마세요. 제목도 그대로 반복하지 말고, snippet에 있는 내용만 근거로 설명하세요.\n"
        "추측하거나 검색 근거 밖의 일정을 만들어내지 마세요.\n\n"
        f"사용자 질문: {keyword}\n\n"
        "출력 형식:\n"
        "1. {1번 source 설명}\n"
        "2. {2번 source 설명}\n"
        "3. {3번 source 설명}\n\n"
        "검색 source:\n"
        + "\n\n".join(source_blocks)
    )


def _parse_link_descriptions(raw_answer: str, expected_count: int) -> list[str] | None:
    if not raw_answer or expected_count <= 0:
        return None

    descriptions_by_index: dict[int, str] = {}
    for line in raw_answer.splitlines():
        match = re.match(r"^\s*(\d+)[.)]\s*(.+?)\s*$", line)
        if not match:
            continue
        index = int(match.group(1))
        if 1 <= index <= expected_count:
            descriptions_by_index[index] = _clean_description(match.group(2))

    if len(descriptions_by_index) == expected_count:
        descriptions = [descriptions_by_index[index] for index in range(1, expected_count + 1)]
        return descriptions if all(descriptions) else None

    lines = [_clean_description(line) for line in raw_answer.splitlines() if line.strip()]
    if expected_count == 1 and len(lines) == 1 and lines[0]:
        return lines

    return None


def _clean_description(description: str) -> str:
    cleaned = re.sub(r"https?://\S+", "", description).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.rstrip()


def _format_link_guide_answer(
    keyword: str,
    links: list[_SourceLink],
    descriptions: list[str],
) -> str:
    if not links:
        return (
            f'"{keyword}"와 관련해 answer에 표시할 공식 링크를 찾지 못했습니다.\n'
            "검색 결과의 전체 source 목록을 확인해 주세요."
        )

    lines = [
        f'"{keyword}"와 관련해 확인할 수 있는 공식 링크를 찾았습니다.',
        "아래 링크들은 검색 결과에서 유사도가 높은 한성대학교 공지입니다.",
    ]
    for index, link in enumerate(links, start=1):
        description = (
            descriptions[index - 1]
            if index - 1 < len(descriptions) and descriptions[index - 1]
            else _default_link_description(link)
        )
        lines.extend(
            [
                "",
                f"{index}. {link.title}",
                f"   {description}",
                "   이동하시려면 아래 링크를 눌러주세요.",
                f"   {link.url}",
            ]
        )
    return "\n".join(lines)


def _default_link_description(link: _SourceLink) -> str:
    return f"{link.title}와 관련된 한성대학교 공식 안내 링크입니다."


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
