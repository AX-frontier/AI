from __future__ import annotations

import html
import re
from datetime import datetime
from zoneinfo import ZoneInfo

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
        location_card = _build_location_card(first)
        answer = (
            f"'{result.keyword}' 검색 결과 {result_count}건을 찾았습니다. "
            f"대표 도서 '{first.title}'의 청구기호는 {first.holdingCallNo or '정보 없음'}이고, "
            f"소장 위치는 {first.stackLocation or '정보 없음'}"
            f"{' ' + first.stackShelf if first.stackShelf else ''}입니다. "
            f"{location_card['guideText']}"
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
        summary=(
            {
                "contentType": "book_location",
                "locationCard": _build_location_card(first),
            }
            if intent == "BOOK_LOCATION"
            else None
        ),
        extractedTables=None,
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
    # GUIDE 답변은 항상 짧은 요약으로 고정하고, 상세/표는 structured 필드로 분리한다.
    answer = _compose_guide_answer(answer_title, answer_content)
    structured_summary = _build_guide_summary(answer_title, answer_content)
    extracted_tables = _extract_markdown_tables(answer_content)
    if structured_summary is not None:
        structured_summary["operatingHours"] = _build_operating_hours(extracted_tables, answer_content)
        structured_summary["currentStatus"] = _build_current_status(
            structured_summary["operatingHours"],
            context.keyword,
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
        summary=structured_summary,
        extractedTables=extracted_tables or [],
    )


def _compose_guide_answer(title: str, content: str) -> str:
    plain = _strip_table_lines(_normalize_content(content))
    snippet = _first_meaningful_line(plain) or "관련 안내를 찾았습니다."
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


def build_guide_response_with_answer(
    intent: LibraryIntent,
    context: GuideContext,
    confidence: float,
    answer: str,
) -> LibraryChatResponse:
    """스트리밍으로 생성된 answer 문자열로 guide 응답을 조립한다."""
    primary_chunk = context.chunks[0] if context.chunks else None
    primary = context.docs[0] if context.docs else None
    raw_content = ""
    raw_title = context.keyword
    if primary_chunk:
        raw_content = primary_chunk.content
        raw_title = primary_chunk.title
    elif primary:
        raw_content = primary.content
        raw_title = primary.title
    sources = _build_guide_sources(context)
    return LibraryChatResponse(
        intent=intent,
        answer=answer.strip() if answer and answer.strip() else _compose_guide_answer(raw_title, raw_content),
        sources=sources,
        confidence=max(confidence, 0.75),
        fallbackUsed=False,
        fallbackReason=None,
        searchKeyword=context.keyword,
        resultCount=0,
        matchedBooks=[],
        summary=_build_streaming_summary(context, raw_title, raw_content),
        extractedTables=_extract_markdown_tables(raw_content) if raw_content else [],
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
        summary=None,
        extractedTables=None,
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


def _normalize_content(content: str) -> str:
    """Guide fallback 답변은 원문 전체를 유지하고 줄바꿈을 보존한다."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                compact_lines.append("")
            previous_blank = True
            continue
        compact_lines.append(line)
        previous_blank = False
    return "\n".join(compact_lines).strip()


def _strip_table_lines(content: str) -> str:
    lines = content.split("\n")
    kept = [line for line in lines if not _is_table_line(line)]
    return "\n".join(kept).strip()


def _first_meaningful_line(content: str) -> str | None:
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            continue
        if stripped in {"---"}:
            continue
        if stripped.startswith("문서 제목:"):
            continue
        if stripped.startswith("섹션:"):
            section = stripped.split(":", 1)[-1].strip()
            if section:
                return f"{section} 안내입니다. 자세한 내용은 표/상세 정보를 확인해 주세요."
            continue
        return stripped
    return None


def _build_guide_summary(title: str, content: str) -> dict[str, object]:
    normalized = _normalize_content(content)
    return {
        "contentType": "library_guide",
        "title": title,
        "content": normalized,
    }


def _build_streaming_summary(
    context: GuideContext,
    raw_title: str,
    raw_content: str,
) -> dict[str, object] | None:
    if not raw_content:
        return None
    summary = _build_guide_summary(raw_title, raw_content)
    keyword = context.keyword
    tables = _extract_markdown_tables(raw_content)
    summary["operatingHours"] = _build_operating_hours(tables, raw_content)
    summary["currentStatus"] = _build_current_status(summary["operatingHours"], keyword)
    return summary


def _extract_markdown_tables(content: str) -> list[dict[str, object]]:
    lines = [line.rstrip() for line in _normalize_content(content).split("\n")]
    tables: list[dict[str, object]] = []
    i = 0
    while i < len(lines):
        if not _is_table_line(lines[i]):
            i += 1
            continue

        block: list[str] = []
        while i < len(lines) and _is_table_line(lines[i]):
            block.append(lines[i].strip())
            i += 1

        table = _parse_table_block(block)
        if table is not None:
            tables.append(table)
    return tables


def _parse_table_block(lines: list[str]) -> dict[str, object] | None:
    if len(lines) < 2:
        return None

    headers = _normalize_row_length(_table_cells(lines[0]))
    data_start = 1
    if _is_table_separator_line(lines[1]):
        data_start = 2

    rows: list[list[str]] = []
    for line in lines[data_start:]:
        if _is_table_separator_line(line):
            continue
        cells = _normalize_row_length(_table_cells(line))
        if cells:
            rows.append(cells)

    if not headers:
        return None
    headers = [header for header in headers if header]
    if not headers:
        return None

    rows = [_trim_or_pad_row(row, len(headers)) for row in rows]
    rows = [row for row in rows if _is_data_row(row, headers)]

    return {"headers": headers, "rows": rows}


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _normalize_row_length(cells: list[str]) -> list[str]:
    normalized = list(cells)
    while normalized and normalized[-1] == "":
        normalized.pop()
    return normalized


def _trim_or_pad_row(row: list[str], target_len: int) -> list[str]:
    if len(row) >= target_len:
        return row[:target_len]
    return [*row, *([""] * (target_len - len(row)))]


def _is_data_row(row: list[str], headers: list[str]) -> bool:
    # 표 내부의 서브헤더 행(예: 학기중/방학중)을 데이터 행에서 제외한다.
    if not row:
        return False
    if len(row) <= 2 and any(token in row[0] for token in ("학기중", "방학중")):
        return False
    if row[0] in {"학기중", "방학중"}:
        return False
    return any(cell for cell in row) and row != headers


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]


def _is_table_separator_line(line: str) -> bool:
    stripped = line.strip().strip("|").strip()
    if not stripped:
        return False
    tokens = [token.strip() for token in stripped.split("|")]
    if not tokens:
        return False
    return all(re.fullmatch(r":?-{3,}:?", token) for token in tokens)


def _build_operating_hours(
    tables: list[dict[str, object]],
    raw_content: str,
) -> list[dict[str, str]]:
    slots = _parse_hours_from_tables(tables)
    if slots:
        return _dedupe_slots(slots)
    return _dedupe_slots(_parse_hours_from_text(raw_content))


def _parse_hours_from_tables(tables: list[dict[str, object]]) -> list[dict[str, str]]:
    slots: list[dict[str, str]] = []
    for table in tables:
        headers = [str(h).strip() for h in table.get("headers", [])]
        rows = table.get("rows", [])
        if not isinstance(rows, list) or len(headers) < 2:
            continue
        for row in rows:
            if not isinstance(row, list):
                continue
            values = [str(v).strip() for v in row]
            if len(values) < 2:
                continue
            location = values[0]
            period_headers = _resolve_period_headers(headers, values)
            for idx in range(1, min(len(values), len(period_headers))):
                period = period_headers[idx]
                value = values[idx]
                if not value:
                    continue
                if not _looks_hours_value(value):
                    continue
                slots.extend(_normalize_time_slot(location, period, value))
    return slots


def _resolve_period_headers(headers: list[str], first_row: list[str]) -> list[str]:
    normalized = [h.strip() for h in headers]
    if len(headers) >= 3 and headers[1] == "구분" and headers[2] == "이용 시간":
        if len(first_row) >= 4 and first_row[0] in {"학기중", "방학중"}:
            normalized = [headers[0], first_row[0], first_row[1], first_row[1] if len(first_row) < 4 else first_row[1]]
        else:
            normalized = [headers[0], "학기중", "학기중", "방학중"]
    return normalized


def _parse_hours_from_text(content: str) -> list[dict[str, str]]:
    slots: list[dict[str, str]] = []
    normalized = _normalize_content(content)
    for line in normalized.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if "휴관" in stripped and ":" not in stripped:
            continue
        for match in re.finditer(r"(평일|토요일|일요일|공휴일)\s*:\s*([^|]+)", stripped):
            day_label = match.group(1)
            value = match.group(2).strip()
            slots.extend(_normalize_time_slot("학술정보관", day_label, value))
    return slots


def _normalize_time_slot(location: str, period: str, raw_value: str) -> list[dict[str, str]]:
    text = raw_value.replace("~", " ~ ").replace("〜", " ~ ")
    text = re.sub(r"\s+", " ", text).strip()
    period_label = period.strip()
    entries: list[dict[str, str]] = []
    marker_pattern = re.compile(
        r"(평\s*일|토\s*요일|일\s*요일|일·공휴일|토·일·공휴일|토·일|공휴일)\s*:\s*"
    )
    matches = list(marker_pattern.finditer(text))
    if matches:
        for idx, match in enumerate(matches):
            label = re.sub(r"\s+", "", match.group(1))
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            value = text[start:end].strip(" -")
            entries.append(_slot(location, period_label, _day_label_to_key(label), value))
        return entries
    return [_slot(location, period_label, _period_to_day_key(period_label), text)]


def _day_label_to_key(label: str) -> str:
    mapping = {
        "평일": "weekday",
        "토요일": "saturday",
        "일요일": "sunday",
        "일·공휴일": "holiday",
        "공휴일": "holiday",
        "토·일·공휴일": "weekend_holiday",
        "토·일": "weekend",
    }
    return mapping.get(label, "general")


def _looks_hours_value(value: str) -> bool:
    if "휴관" in value:
        return True
    if re.search(r"\d{1,2}:\d{2}\s*~\s*\d{1,2}:\d{2}", value):
        return True
    if any(token in value for token in ("평 일", "평일", "토요일", "일요일", "공휴일")):
        return True
    return False


def _dedupe_slots(slots: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[dict[str, str]] = []
    for slot in slots:
        key = (
            slot.get("location", ""),
            slot.get("period", ""),
            slot.get("dayType", ""),
            slot.get("start", ""),
            slot.get("end", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(slot)
    return unique


def _period_to_day_key(period: str) -> str:
    lowered = period.lower()
    if "학기중" in lowered:
        return "semester"
    if "방학중" in lowered:
        return "vacation"
    if "토요일" in lowered:
        return "saturday"
    if "일요일" in lowered:
        return "sunday"
    if "공휴일" in lowered:
        return "holiday"
    return "general"


def _slot(location: str, period: str, day_key: str, value: str) -> dict[str, str]:
    status = "closed" if ("휴관" in value and not _extract_time_range(value)[0]) else "open"
    start, end = _extract_time_range(value)
    return {
        "location": location,
        "period": period,
        "dayType": day_key,
        "hoursText": value,
        "status": status,
        "start": start or "",
        "end": end or "",
    }


def _extract_time_range(text: str) -> tuple[str | None, str | None]:
    match = re.search(r"(\d{1,2}:\d{2})\s*~\s*(\d{1,2}:\d{2})", text)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _build_current_status(
    operating_hours: list[dict[str, str]],
    keyword: str,
) -> dict[str, object] | None:
    if not operating_hours:
        return None
    if not _is_time_question(keyword):
        return None

    now = datetime.now(ZoneInfo("Asia/Seoul"))
    day_key = _day_key(now.weekday())
    now_minutes = now.hour * 60 + now.minute

    primary_types = {day_key}
    fallback_types = {"semester", "vacation", "general"}
    candidates = [
        slot
        for slot in operating_hours
        if slot.get("dayType") in primary_types
        and slot.get("status") == "open"
        and slot.get("start")
        and slot.get("end")
    ]
    if not candidates:
        candidates = [
            slot
            for slot in operating_hours
            if slot.get("dayType") in fallback_types
            and slot.get("status") == "open"
            and slot.get("start")
            and slot.get("end")
        ]
    open_now = False
    closes_at = None
    for slot in candidates:
        start = _to_minutes(slot["start"])
        end = _to_minutes(slot["end"])
        if start is None or end is None:
            continue
        if start <= now_minutes <= end:
            open_now = True
            closes_at = slot["end"]
            break

    return {
        "timezone": "Asia/Seoul",
        "checkedAt": now.isoformat(),
        "isOpenNow": open_now,
        "closesAt": closes_at,
    }


def _to_minutes(hhmm: str) -> int | None:
    try:
        hour, minute = hhmm.split(":")
        return int(hour) * 60 + int(minute)
    except Exception:
        return None


def _day_key(weekday: int) -> str:
    if weekday <= 4:
        return "weekday"
    if weekday == 5:
        return "saturday"
    return "sunday"


def _is_time_question(keyword: str) -> bool:
    hints = ("오늘", "지금", "몇 시", "열어", "열어요", "운영", "시간", "언제")
    lowered = keyword.lower()
    return any(hint in lowered for hint in hints)


def _build_location_card(book: MatchedBook) -> dict[str, str]:
    location = book.stackLocation or "위치 정보 없음"
    shelf = book.stackShelf or "서가 정보 없음"
    call_no = book.holdingCallNo or "청구기호 정보 없음"
    floor = _infer_floor_label(location)
    return {
        "title": book.title,
        "floor": floor,
        "location": location,
        "shelfCode": shelf,
        "callNo": call_no,
        "guideText": (
            f"{location} {shelf} 서가에서 찾을 수 있습니다. "
            "서가 코드는 자료실 내 책장 위치 코드이며, 안내데스크에 보여주면 빠르게 안내받을 수 있습니다."
        ),
    }


def _infer_floor_label(location: str) -> str:
    direct = re.search(r"(\d+\s*F)", location, flags=re.IGNORECASE)
    if direct:
        return direct.group(1).replace(" ", "").upper()
    floor_match = re.search(r"(\d+)층", location)
    if floor_match:
        return f"{floor_match.group(1)}층"
    return "층 정보 없음"
