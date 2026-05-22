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
STRICT_MATCH_CHECK_TOP_N = 3
SCHOLARSHIP_ANCHOR_TERMS = ("장학", "scholarship")
SCHOLARSHIP_TITLE_TERMS = ("장학", "장학금", "장학생", "scholarship")
SCHOLARSHIP_CATEGORY_TERMS = ("장학공지",)
MAIN_LINK_DISCLAIMER_TERMS = (
    "관련 없음",
    "관련없음",
    "관련은 없",
    "관련이 없",
    "관련 없",
    "관련 없는",
    "관련성이 낮",
    "무관",
    "직접 관련은 없",
    "직접 관련이 없",
    "직접 관련 없",
    "직접 관련된 것은 아니",
    "직접적인 관련은 없",
    "직접적인 관련이 없",
)
MAIN_SOURCE_RELEVANCE_STOP_TERMS = {
    "공지",
    "안내",
    "관련",
    "링크",
    "url",
    "페이지",
    "학교",
    "한성",
    "한성대",
    "한성대학교",
    "신청",
    "기간",
    "대상",
    "방법",
    "정보",
    "보고싶어",
    "보고싶은데",
    "보고싶습니다",
    "알고싶어",
    "알고싶은데",
    "찾아보고싶어",
    "찾아보고싶음",
}


@dataclass(frozen=True)
class _SourceLink:
    title: str
    url: str
    category: str | None
    posted_date: str | None
    snippet: str


@dataclass(frozen=True)
class _SourceRelevanceGroup:
    aliases: tuple[str, ...]
    token_only: bool


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

    display_chunks = filter_relevant_source_chunks(result.keyword, result.chunks)
    if not display_chunks:
        return build_main_fallback_response(
            result.keyword,
            "질문과 직접적으로 일치하는 공지 데이터를 찾지 못했습니다. 관련 데이터가 준비 중일 수 있습니다.",
            reason_code="TOPIC_MISMATCH_NO_DATA",
        )

    links = select_source_links(display_chunks)
    if not links:
        return build_main_fallback_response(
            result.keyword,
            "질문과 직접적으로 일치하는 공식 링크를 찾지 못했습니다. 관련 데이터가 준비 중일 수 있습니다.",
            reason_code="TOPIC_MISMATCH_NO_DATA",
        )
    if _fails_strict_match_filter(result.keyword, links):
        return build_main_fallback_response(
            result.keyword,
            "질문과 직접적으로 일치하는 공지 데이터를 찾지 못했습니다. 관련 데이터가 준비 중일 수 있습니다.",
            reason_code="TOPIC_MISMATCH_NO_DATA",
        )

    descriptions = _generate_link_descriptions(result.keyword, llm_client, links)
    links, descriptions = filter_disclaimed_source_links(links, descriptions)
    source_chunks = source_chunks_for_links(display_chunks, links)
    if not links or not source_chunks:
        return build_main_fallback_response(
            result.keyword,
            "질문과 직접적으로 일치하는 공식 링크를 찾지 못했습니다. 관련 데이터가 준비 중일 수 있습니다.",
            reason_code="TOPIC_MISMATCH_NO_DATA",
        )

    answer = _format_link_guide_answer(result.keyword, links, descriptions)
    sources = [_chunk_to_source(chunk) for chunk in source_chunks]
    display_primary = source_chunks[0]
    confidence = max(0.5, min(0.95, display_primary.score))
    return MainChatResponse(
        intent=_infer_main_intent(display_primary),
        answer=answer,
        sources=sources,
        confidence=round(confidence, 3),
        fallbackUsed=False,
        fallbackReason=None,
        searchKeyword=result.keyword,
        resultCount=len(source_chunks),
    )


def build_main_fallback_response(
    keyword: str,
    reason: str,
    *,
    reason_code: str = "GENERIC",
    related_chunks: list[MainChunkRecord] | None = None,
) -> MainChatResponse:
    if reason_code == "LOW_SIMILARITY":
        answer = (
            "질문을 조금 더 구체적으로 다시 입력해 주세요. "
            "예: 장학금 종류, 신청 기간, 대상, 학기 등을 함께 적어주시면 더 정확히 안내할 수 있습니다."
        )
    elif reason_code == "NO_CHUNKS":
        answer = (
            "관련 공지를 찾지 못했습니다. "
            "찾고 싶은 주제(예: 장학금/수강신청/휴복학)와 기간을 포함해 다시 질문해 주세요."
        )
    elif reason_code in {"TOPIC_MISMATCH_NO_DATA", "GENERIC"} and "관련 데이터가 준비 중" in reason:
        answer = _build_data_preparing_answer(keyword, related_chunks or [])
    else:
        answer = f"{reason} 공식 공지나 학사 안내 데이터가 적재된 뒤 다시 확인해 주세요."
    return MainChatResponse(
        intent="MAIN_GENERAL",
        answer=answer,
        sources=[],
        confidence=0.35,
        fallbackUsed=True,
        fallbackReason=reason,
        fallbackReasonCode=reason_code,
        searchKeyword=keyword,
        resultCount=0,
    )


def _compose_answer(result: MainSearchResult, *, links: list[_SourceLink] | None = None) -> str:
    links = links if links is not None else select_source_links(result.chunks)
    descriptions = _generate_link_descriptions(result.keyword, None, links)
    links, descriptions = filter_disclaimed_source_links(links, descriptions)
    return _format_link_guide_answer(result.keyword, links, descriptions)


def _generate_answer(
    result: MainSearchResult,
    llm_client: LLMClient | None,
    *,
    links: list[_SourceLink] | None = None,
) -> str:
    if llm_client is None:
        return _compose_answer(result, links=links)

    links = links if links is not None else select_source_links(result.chunks)
    descriptions = _generate_link_descriptions(result.keyword, llm_client, links)
    links, descriptions = filter_disclaimed_source_links(links, descriptions)
    return _format_link_guide_answer(result.keyword, links, descriptions)


def _generate_link_descriptions(
    keyword: str,
    llm_client: LLMClient | None,
    links: list[_SourceLink],
) -> list[str]:
    if llm_client is None:
        return [_default_link_description(link) for link in links]

    if not links:
        return []

    try:
        raw_answer = llm_client.generate(_build_link_description_prompt(keyword, links))
    except Exception:
        return [_default_link_description(link) for link in links]

    descriptions = _parse_link_descriptions(raw_answer, len(links))
    if descriptions is None:
        descriptions = [_default_link_description(link) for link in links]
    return descriptions


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


def source_chunks_for_links(
    chunks: list[MainChunkRecord],
    links: list[_SourceLink],
) -> list[MainChunkRecord]:
    """최종 노출 링크와 동일한 chunk만 sources에 남긴다."""
    first_chunk_by_url: dict[str, MainChunkRecord] = {}
    for chunk in chunks:
        if chunk.url and chunk.url not in first_chunk_by_url:
            first_chunk_by_url[chunk.url] = chunk
    return [first_chunk_by_url[link.url] for link in links if link.url in first_chunk_by_url]


def filter_disclaimed_source_links(
    links: list[_SourceLink],
    descriptions: list[str],
) -> tuple[list[_SourceLink], list[str]]:
    kept_links: list[_SourceLink] = []
    kept_descriptions: list[str] = []
    for index, link in enumerate(links):
        description = (
            descriptions[index]
            if index < len(descriptions) and descriptions[index]
            else _default_link_description(link)
        )
        if _is_disclaimed_link_description(description):
            continue
        kept_links.append(link)
        kept_descriptions.append(description)
    return kept_links, kept_descriptions


def filter_relevant_source_chunks(keyword: str, chunks: list[MainChunkRecord]) -> list[MainChunkRecord]:
    """답변과 sources에 노출할 chunk를 질문 핵심어 기준으로 제한한다."""
    groups = _extract_source_relevance_groups(keyword)
    if not groups:
        return chunks
    return [chunk for chunk in chunks if _matches_source_relevance_groups(groups, chunk)]


def _extract_source_relevance_groups(keyword: str) -> list[_SourceRelevanceGroup]:
    raw_terms = re.findall(r"[0-9A-Za-z가-힣]+", (keyword or "").lower())
    groups: list[_SourceRelevanceGroup] = []
    for raw_term in raw_terms:
        term = _strip_source_relevance_suffix(_strip_korean_particle(raw_term))
        if len(term) < 2 or term.isdigit() or term in MAIN_SOURCE_RELEVANCE_STOP_TERMS:
            continue
        aliases = _dedupe_terms(_expand_source_relevance_term(term))
        aliases = [
            alias
            for alias in aliases
            if len(alias) >= 2 and alias not in MAIN_SOURCE_RELEVANCE_STOP_TERMS
        ]
        if aliases:
            groups.append(
                _SourceRelevanceGroup(
                    aliases=tuple(aliases),
                    token_only=_is_short_latin_term(term),
                )
            )

    deduped: list[_SourceRelevanceGroup] = []
    seen: set[tuple[tuple[str, ...], bool]] = set()
    for group in groups:
        key = (group.aliases, group.token_only)
        if key in seen:
            continue
        deduped.append(group)
        seen.add(key)
    return deduped


def _strip_source_relevance_suffix(term: str) -> str:
    for suffix in ("관련", "공지", "안내", "정보"):
        if len(term) > len(suffix) + 1 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def _expand_source_relevance_term(term: str) -> list[str]:
    expanded = [term]
    if "전공" in term:
        expanded.append("전공")
    if "복수" in term:
        expanded.append("복수")
    if "부전공" in term:
        expanded.append("부전공")
    if "수강" in term:
        expanded.append("수강")
    if "휴복학" in term:
        expanded.extend(["휴학", "복학"])
    if "장학" in term:
        expanded.append("장학")
    if "프론티어" in term or "프런티어" in term:
        expanded.extend(["프론티어", "프런티어", "frontier", "prontier"])
    if "도서관" in term:
        expanded.extend(["도서관", "학술정보관"])
    if "학술정보관" in term:
        expanded.extend(["학술정보관", "도서관"])
    return expanded


def _matches_source_relevance_groups(
    groups: list[_SourceRelevanceGroup],
    chunk: MainChunkRecord,
) -> bool:
    source_text = " ".join(
        [
            chunk.title or "",
            chunk.text or "",
            chunk.category or "",
        ]
    )
    normalized_haystack = _normalize_source_text(source_text)
    tokens = set(_source_match_tokens(source_text))
    return all(
        _matches_source_relevance_group(group, normalized_haystack, tokens)
        for group in groups
    )


def _matches_source_relevance_group(
    group: _SourceRelevanceGroup,
    normalized_haystack: str,
    tokens: set[str],
) -> bool:
    for alias in group.aliases:
        normalized_alias = _normalize_source_text(alias)
        if not normalized_alias:
            continue
        if group.token_only or _is_short_latin_term(alias):
            if normalized_alias in tokens:
                return True
            continue
        if normalized_alias in normalized_haystack:
            return True
    return False


def _source_match_tokens(value: str) -> list[str]:
    return [
        _normalize_source_text(token)
        for token in re.findall(r"[0-9A-Za-z가-힣]+", (value or "").lower())
        if _normalize_source_text(token)
    ]


def _normalize_source_text(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", (value or "").lower())


def _is_short_latin_term(term: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9]{2,3}", (term or "").lower()))


def _dedupe_terms(terms: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        deduped.append(term)
        seen.add(term)
    return deduped


def _is_disclaimed_link_description(description: str) -> bool:
    normalized = re.sub(r"\s+", " ", (description or "").lower())
    compact = _normalize_source_text(description)
    return any(term in normalized or _normalize_source_text(term) in compact for term in MAIN_LINK_DISCLAIMER_TERMS)


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


def _fails_strict_match_filter(keyword: str, links: list[_SourceLink]) -> bool:
    if not links:
        return False
    if _is_scholarship_query(keyword):
        top_links = links[:STRICT_MATCH_CHECK_TOP_N]
        return not any(_is_scholarship_link_match(link) for link in top_links)
    return False


def _is_scholarship_query(keyword: str) -> bool:
    normalized = (keyword or "").lower()
    return any(term in normalized for term in SCHOLARSHIP_ANCHOR_TERMS)


def _is_scholarship_link_match(link: _SourceLink) -> bool:
    title = (link.title or "").lower()
    category = (link.category or "").lower()
    if any(term in category for term in SCHOLARSHIP_CATEGORY_TERMS):
        return True
    return any(term in title for term in SCHOLARSHIP_TITLE_TERMS)


def _build_data_preparing_answer(keyword: str, chunks: list[MainChunkRecord]) -> str:
    query_focus = _summarize_query_focus(keyword)
    related = _top_related_references(chunks, limit=2)
    if not related:
        return (
            "데이터 준비중입니다. "
            f'"{query_focus}"와 직접 일치하는 공지를 아직 찾지 못했습니다. '
            "관련 데이터를 추가 수집해 안내드리겠습니다."
        )

    lines = [
        "데이터 준비중입니다.",
        f'"{query_focus}"에 대해 직접 일치하는 공지는 아직 확인되지 않았습니다.',
        "대신 참고하기 좋은 관련 공지를 먼저 추천드립니다.",
    ]
    for index, (title, url) in enumerate(related, start=1):
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   {url}")
    lines.append(f'"{query_focus}" 관련 공지를 추가 수집해 업데이트해드리겠습니다.')
    return "\n".join(lines)


def _summarize_query_focus(keyword: str) -> str:
    raw_terms = re.findall(r"[0-9A-Za-z가-힣]+", (keyword or "").lower())
    stop_terms = {
        "관련",
        "대해",
        "알고싶은데",
        "알고싶어",
        "알고",
        "싶어",
        "싶은데",
        "찾아보고",
        "찾아보고싶음",
        "찾아보고싶어요",
        "찾아보고",
        "싶습니다",
        "알려줘",
        "공지",
    }
    compact: list[str] = []
    for term in raw_terms:
        term = _strip_korean_particle(term)
        if len(term) < 2 or term in stop_terms:
            continue
        if term not in compact:
            compact.append(term)
    if compact:
        return " ".join(compact[:3])
    return (keyword or "요청 주제").strip()


def _strip_korean_particle(term: str) -> str:
    particles = (
        "으로",
        "에서",
        "까지",
        "부터",
        "에게",
        "한테",
        "처럼",
        "보다",
        "에서",
        "으로",
        "과",
        "와",
        "의",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "도",
        "로",
    )
    for particle in particles:
        if len(term) > len(particle) + 1 and term.endswith(particle):
            return term[: -len(particle)]
    return term


def _top_related_references(chunks: list[MainChunkRecord], limit: int = 2) -> list[tuple[str, str | None]]:
    refs: list[tuple[str, str | None]] = []
    seen_titles: set[str] = set()
    for chunk in chunks:
        title = (chunk.title or "").strip()
        if not title or title in seen_titles:
            continue
        refs.append((title, chunk.url))
        seen_titles.add(title)
        if len(refs) >= limit:
            break
    return refs
