from __future__ import annotations

import re
from dataclasses import dataclass

from agents.main_agent.api.schemas import MainChatResponse, MainSource
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.models import MainChunkRecord
from agents.main_agent.retrieval import MainSearchResult

MAIN_VECTOR_THRESHOLD = 0.45
MAIN_TOPIC_MISMATCH_THRESHOLD = 0.34
MAIN_OPERATIONAL_HELP_TERMS = (
    "잃어버",
    "분실",
    "전화",
    "연락",
    "문의처",
    "어디에",
    "어디로",
)
MAIN_LOST_ITEM_QUERY_TERMS = ("잃어버", "분실", "유실", "습득")
MAIN_LOST_ITEM_EVIDENCE_TERMS = (
    "분실",
    "유실",
    "습득",
    "유실물",
    "lost",
    "found",
    "인포메이션데스크",
    "학생처",
    "총무",
    "민원",
)
MAIN_LINK_AMBIGUOUS_ABS_GAP_THRESHOLD = 0.05
MAIN_LINK_AMBIGUOUS_RATIO_THRESHOLD = 1.08
MAIN_LINK_SINGLE_TOP1_THRESHOLD = 0.85
MAIN_LINK_DEFAULT_LIMIT = 3


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
        return build_main_fallback_response(
            result.keyword,
            "관련 학교 공지 또는 안내 chunk를 찾지 못했습니다.",
            reason_code="NO_CHUNKS",
        )

    primary = result.chunks[0]
    if primary.score < MAIN_VECTOR_THRESHOLD:
        return build_main_fallback_response(
            result.keyword,
            "관련 학교 공지 또는 안내 chunk의 유사도가 낮습니다.",
            reason_code="LOW_SIMILARITY",
        )
    if _looks_topic_mismatch(result.keyword, result.chunks):
        return build_main_fallback_response(
            result.keyword,
            "질문과 직접적으로 일치하는 공지 데이터를 찾지 못했습니다. 관련 데이터가 준비 중일 수 있습니다.",
            reason_code="TOPIC_MISMATCH_NO_DATA",
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


def build_main_fallback_response(
    keyword: str,
    reason: str,
    *,
    reason_code: str = "GENERIC",
) -> MainChatResponse:
    return MainChatResponse(
        intent="MAIN_GENERAL",
        answer=f"{reason} 공식 공지나 학사 안내 데이터가 적재된 뒤 다시 확인해 주세요.",
        sources=[],
        confidence=0.35,
        fallbackUsed=True,
        fallbackReason=reason,
        fallbackReasonCode=reason_code,
        searchKeyword=keyword,
        resultCount=0,
    )


def _compose_answer(result: MainSearchResult) -> str:
    links = select_source_links(result.chunks)
    descriptions = [_default_link_description(link) for link in links]
    return _format_link_guide_answer(result.keyword, links, descriptions)


def _generate_answer(result: MainSearchResult, llm_client: LLMClient | None) -> str:
    if llm_client is None:
        return _compose_answer(result)

    links = select_source_links(result.chunks)
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
    if limit <= 0:
        return []
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


def select_source_links(chunks: list[MainChunkRecord]) -> list[_SourceLink]:
    limit = _decide_link_limit(chunks)
    return _top_source_links(chunks, limit=limit)


def _decide_link_limit(chunks: list[MainChunkRecord]) -> int:
    if not chunks:
        return 0
    if len(chunks) < 2:
        return 1
    top1 = chunks[0].score
    top2 = chunks[1].score
    if top2 <= 0:
        return 1 if top1 >= MAIN_LINK_SINGLE_TOP1_THRESHOLD else MAIN_LINK_DEFAULT_LIMIT

    abs_gap = top1 - top2
    ratio = top1 / top2
    is_ambiguous_boundary = (
        abs_gap < MAIN_LINK_AMBIGUOUS_ABS_GAP_THRESHOLD
        or ratio < MAIN_LINK_AMBIGUOUS_RATIO_THRESHOLD
    )
    if is_ambiguous_boundary:
        return MAIN_LINK_DEFAULT_LIMIT

    if top1 >= MAIN_LINK_SINGLE_TOP1_THRESHOLD:
        return 1
    return MAIN_LINK_DEFAULT_LIMIT


def _build_link_description_prompt(keyword: str, links: list[_SourceLink]) -> str:
    if len(links) == 1:
        return _build_single_link_description_prompt(keyword, links[0])
    return _build_multi_link_description_prompt(keyword, links)


def _build_multi_link_description_prompt(keyword: str, links: list[_SourceLink]) -> str:
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


def _build_single_link_description_prompt(keyword: str, link: _SourceLink) -> str:
    return (
        "당신은 한성대학교 공식 공지와 학사 안내 검색 결과를 설명하는 AI입니다.\n"
        "아래 단일 source를 바탕으로 한국어 3~4문장으로 상세 안내하세요.\n"
        "핵심 내용, 대상, 일정(있으면), 유의사항(있으면)을 포함하고 검색 근거 밖의 추측은 금지합니다.\n"
        "URL은 절대 출력하지 마세요.\n\n"
        f"사용자 질문: {keyword}\n\n"
        "검색 source:\n"
        f"title: {link.title}\n"
        f"url: {link.url}\n"
        f"category: {link.category or '정보 없음'}\n"
        f"posted_date: {link.posted_date or '정보 없음'}\n"
        f"snippet: {link.snippet}"
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
    if expected_count == 1 and lines:
        merged = " ".join(line for line in lines if line)
        return [merged] if merged else None

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

    if len(links) == 1:
        link = links[0]
        description = descriptions[0] if descriptions and descriptions[0] else _default_link_description(link)
        return "\n".join(
            [
                f'"{keyword}"와 관련해 확인할 수 있는 공식 링크를 찾았습니다.',
                "검색 결과에서 관련도가 가장 높은 대표 링크 1건을 안내드립니다.",
                "",
                f"1. {link.title}",
                f"   {description}",
                "   이동하시려면 아래 링크를 눌러주세요.",
                f"   {link.url}",
            ]
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
    return (
        f"{link.title}와 관련된 한성대학교 공식 안내 링크입니다. "
        "핵심 일정과 대상, 신청 또는 확인 절차를 본문 기준으로 확인하는 데 도움이 됩니다."
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


def _looks_topic_mismatch(keyword: str, chunks: list[MainChunkRecord]) -> bool:
    if not chunks:
        return False
    if not _looks_operational_help_query(keyword):
        return False
    if _looks_lost_item_query(keyword) and not _has_lost_item_evidence(chunks[:3]):
        return True
    terms = _extract_query_terms(keyword)
    if not terms:
        return False
    top_chunks = chunks[:3]
    matched_terms: set[str] = set()
    for term in terms:
        if any(_term_in_chunk(term, chunk) for chunk in top_chunks):
            matched_terms.add(term)
    coverage = len(matched_terms) / len(terms)
    return coverage < MAIN_TOPIC_MISMATCH_THRESHOLD


def _looks_operational_help_query(keyword: str) -> bool:
    normalized = (keyword or "").lower()
    return any(term in normalized for term in MAIN_OPERATIONAL_HELP_TERMS)


def _extract_query_terms(keyword: str) -> list[str]:
    raw_terms = re.findall(r"[0-9A-Za-z가-힣]+", keyword.lower())
    stop_terms = {
        "오늘",
        "지금",
        "뭐",
        "뭐가",
        "있어",
        "있나요",
        "알려줘",
        "문의",
        "공지",
        "사항",
        "어디",
        "전화",
        "방법",
    }
    terms: list[str] = []
    for term in raw_terms:
        if len(term) < 2:
            continue
        if term in stop_terms:
            continue
        terms.append(term)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def _term_in_chunk(term: str, chunk: MainChunkRecord) -> bool:
    title = (chunk.title or "").lower()
    text = (chunk.text or "").lower()
    category = (chunk.category or "").lower()
    return term in title or term in text or term in category


def _looks_lost_item_query(keyword: str) -> bool:
    normalized = (keyword or "").lower()
    return any(term in normalized for term in MAIN_LOST_ITEM_QUERY_TERMS)


def _has_lost_item_evidence(chunks: list[MainChunkRecord]) -> bool:
    for chunk in chunks:
        title = (chunk.title or "").lower()
        text = (chunk.text or "").lower()
        category = (chunk.category or "").lower()
        combined = f"{title} {text} {category}"
        if any(term in combined for term in MAIN_LOST_ITEM_EVIDENCE_TERMS):
            return True
    return False
