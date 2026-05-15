"""전자결재 검토 매뉴얼 기반 자동 규칙.

구현 범위는 날짜/시간/문장부호/항목 번호/붙임/끝표시 같은 텍스트 규칙과,
수입 정산·소요예산 표의 필수 항목 및 금액 일치 여부 확인으로 제한한다.
표는 자동 수정하지 않고 원본 전자결재/HWP 표에 사람이 반영할 코멘트만 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from bs4 import BeautifulSoup

from agents.document_review.api.schemas import ExtractedTable
from agents.document_review.models import (
    CheckRequiredItem,
    DocumentLine,
    FormatNoticeItem,
    RuleFinding,
    TableCheckItem,
)

AMOUNT_TOLERANCE_WON = 1
SMALL_AMOUNT_DIFF_WON = 1000
ITEM_MARKER_PATTERN = re.compile(
    r"^(?P<indent>[ \t\u00a0\u3000]*)(?P<marker>\d+\.|[가-힣]\.|\d+\)|[가-힣]\)|\(\d+\)|\([가-힣]\)|[①-⑳]|[㉮-㉻])(?P<spaces>[^\S\r\n]*)(?P<content>\S.*)$"
)
ITEM_STYLE_ORDER = (
    "decimal_dot",
    "korean_dot",
    "decimal_paren",
    "korean_paren",
    "decimal_wrapped",
    "korean_wrapped",
    "circled_digit",
    "circled_korean",
)
ITEM_STYLE_LABELS = {
    "decimal_dot": "첫째 항목",
    "korean_dot": "둘째 항목",
    "decimal_paren": "셋째 항목",
    "korean_paren": "넷째 항목",
    "decimal_wrapped": "다섯째 항목",
    "korean_wrapped": "여섯째 항목",
    "circled_digit": "일곱째 항목",
    "circled_korean": "여덟째 항목",
}
ITEM_STYLE_EXAMPLES = {
    "decimal_dot": "1., 2., 3.",
    "korean_dot": "가., 나., 다.",
    "decimal_paren": "1), 2), 3)",
    "korean_paren": "가), 나), 다)",
    "decimal_wrapped": "(1), (2), (3)",
    "korean_wrapped": "(가), (나), (다)",
    "circled_digit": "①, ②, ③",
    "circled_korean": "㉮, ㉯, ㉰",
}


@dataclass(frozen=True, slots=True)
class ParsedItemMarker:
    line: DocumentLine
    marker: str
    style: str
    indent_width: int


def apply_safe_suggestions_to_html(body_html: str | None, findings: list[RuleFinding]) -> str | None:
    if not body_html:
        return None
    revised = body_html
    for finding in findings:
        if not finding.editable or not finding.suggested_text:
            continue
        if finding.rule_code == "ATTACHMENT_LABEL":
            revised = _apply_attachment_label_to_html(revised)
            continue
        if finding.rule_code == "ITEM_MARKER_STYLE":
            revised = _apply_item_marker_style_to_html(revised, finding)
            continue
        revised = revised.replace(finding.original_text, finding.suggested_text, 1)
    return revised


def _apply_attachment_label_to_html(body_html: str) -> str:
    soup = BeautifulSoup(body_html, "html.parser")
    for text_node in soup.find_all(string=True):
        original = str(text_node)
        revised = re.sub(
            r"^([ \t\u00a0\u3000]*)첨부(?=[ \t\u00a0\u3000]*\d+\.)",
            r"\1붙임",
            original,
            count=1,
        )
        if revised != original:
            text_node.replace_with(revised)
            return str(soup)
    return body_html


def _apply_item_marker_style_to_html(body_html: str, finding: RuleFinding) -> str:
    if finding.suggested_text is None:
        return body_html
    if finding.original_text in body_html:
        return body_html.replace(finding.original_text, finding.suggested_text, 1)

    original_match = ITEM_MARKER_PATTERN.match(finding.original_text)
    suggested_match = ITEM_MARKER_PATTERN.match(finding.suggested_text)
    if not original_match or not suggested_match:
        return body_html

    original_marker = original_match.group("marker")
    suggested_marker = suggested_match.group("marker")
    soup = BeautifulSoup(body_html, "html.parser")
    marker_pattern = re.compile(rf"^([ \t\u00a0\u3000]*){re.escape(original_marker)}(?=[^\S\r\n]*\S)")
    for text_node in soup.find_all(string=True):
        original = str(text_node)
        revised = marker_pattern.sub(lambda match: f"{match.group(1)}{suggested_marker}", original, count=1)
        if revised != original:
            text_node.replace_with(revised)
            return str(soup)
    return body_html


def split_document_lines(text: str) -> list[DocumentLine]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [
        DocumentLine(number=index, text=line.rstrip())
        for index, line in enumerate(normalized.split("\n"), start=1)
    ]


def review_rules(
    text: str,
    extracted_tables: list[ExtractedTable] | None = None,
) -> tuple[list[RuleFinding], list[CheckRequiredItem], list[FormatNoticeItem]]:
    lines = split_document_lines(text)
    tables = extracted_tables or []
    findings: list[RuleFinding] = []
    checks: list[CheckRequiredItem] = []
    for line in lines:
        findings.extend(_review_line(line))
        checks.extend(_review_amounts(line))

    findings.extend(_review_document_level(lines))
    findings.extend(_review_single_item_sections(lines))
    findings.extend(_review_item_marker_styles(lines))
    checks.extend(_review_related_documents(lines))
    checks.extend(_review_law_references(lines))
    checks.extend(_review_budget_tables(lines, tables))
    checks.extend(_review_attachment_list(lines))
    checks.extend(_review_item_marker_hierarchy(lines))
    format_notices = [
        FormatNoticeItem(
            category="서식",
            message=(
                "텍스트 입력만으로 글꼴, 글자 크기, 줄간격은 자동 확인이 어렵습니다. "
                "전자결재 화면에서 굴림 11pt, 줄간격 180% 여부를 참고 확인해 주세요."
            ),
        )
    ]
    if _has_nested_items(lines):
        format_notices.append(
            FormatNoticeItem(
                category="항목 번호 체계",
                message=(
                    "하위 항목 들여쓰기와 두 줄 이상 항목의 둘째 줄 정렬은 텍스트만으로 자동 판정하기 어렵습니다. "
                    "전자결재 화면에서 상위 항목보다 2타씩 들여쓰기 되었는지 확인해 주세요."
                ),
            )
        )
    return findings, checks, format_notices


def apply_safe_suggestions(text: str, findings: list[RuleFinding]) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for finding in findings:
        if not finding.editable or not finding.suggested_text:
            continue
        if finding.line_start != finding.line_end:
            continue
        index = finding.line_start - 1
        if index < 0 or index >= len(lines):
            continue
        lines[index] = lines[index].replace(finding.original_text, finding.suggested_text, 1)
    return "\n".join(lines)


def _review_line(line: DocumentLine) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    findings.extend(_review_hyphen_dates(line))
    findings.extend(_review_weekday_dates(line))
    findings.extend(_review_dot_dates(line))
    findings.extend(_review_tilde(line))
    findings.extend(_review_colon(line))
    findings.extend(_review_time(line))
    findings.extend(_review_item_spacing(line))
    findings.extend(_review_approval_phrase(line))
    findings.extend(_review_end_marker_spacing(line))
    findings.extend(_review_attachment_spacing(line))
    return findings


def _review_hyphen_dates(line: DocumentLine) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    for match in re.finditer(r"\b(20\d{2})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b", line.text):
        year, month, day = match.groups()
        suggested = f"{year}. {int(month)}. {int(day)}."
        findings.append(
            RuleFinding(
                rule_code="DATE_FORMAT",
                category="날짜 표기",
                severity="MEDIUM",
                status="REVISION_REQUIRED",
                line_start=line.number,
                line_end=line.number,
                original_text=match.group(0),
                suggested_text=suggested,
                reason="날짜는 연, 월, 일 뒤에 마침표를 찍고 마침표 뒤를 띄어 써야 합니다.",
            )
        )
    return findings


def _review_dot_dates(line: DocumentLine) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    pattern = r"\b(20\d{2})\.(\s*)(0?[1-9]|1[0-2])\.(\s*)(0?[1-9]|[12]\d|3[01])\."
    for match in re.finditer(pattern, line.text):
        year, _, month, _, day = match.groups()
        suggested = f"{year}. {int(month)}. {int(day)}."
        if match.group(0) == suggested:
            continue
        findings.append(
            RuleFinding(
                rule_code="DATE_FORMAT",
                category="날짜 표기",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=line.number,
                line_end=line.number,
                original_text=match.group(0),
                suggested_text=suggested,
                reason="월·일 앞의 0을 쓰지 않고, 마침표 뒤에는 한 칸을 띄어야 합니다.",
            )
        )
    return findings


def _review_weekday_dates(line: DocumentLine) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    pattern = r"\b(20\d{2})\.(\s*)(0?[1-9]|1[0-2])\.(\s*)(0?[1-9]|[12]\d|3[01])(\.)?(\s*)\(([월화수목금토일])\)"
    for match in re.finditer(pattern, line.text):
        year, _, month, _, day, day_dot, before_weekday, weekday = match.groups()
        suggested = f"{year}. {int(month)}. {int(day)}.({weekday})"
        if match.group(0) == suggested:
            continue
        reason = "요일을 표시할 경우 날짜 뒤에 붙여 쓰고, 일 뒤에는 마침표를 찍습니다."
        if day_dot and before_weekday:
            reason = "요일은 날짜 뒤에 붙여 쓰며, 날짜와 요일 사이를 띄지 않습니다."
        findings.append(
            RuleFinding(
                rule_code="DATE_FORMAT",
                category="날짜 표기",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=line.number,
                line_end=line.number,
                original_text=match.group(0),
                suggested_text=suggested,
                reason=reason,
            )
        )
    return findings


def _review_tilde(line: DocumentLine) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    for match in re.finditer(r"\S+\s+~\s+\S+", line.text):
        original = match.group(0)
        suggested = re.sub(r"\s*~\s*", "~", original)
        if original == suggested:
            continue
        findings.append(
            RuleFinding(
                rule_code="TILDE_SPACING",
                category="문장부호와 기호",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=line.number,
                line_end=line.number,
                original_text=original,
                suggested_text=suggested,
                reason="기간을 표시하는 물결표(~)는 앞말과 뒷말에 붙여 씁니다.",
            )
        )
    return findings


def _review_colon(line: DocumentLine) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    for match in re.finditer(r"(?<!\d)([가-힣A-Za-z][^:\n]{0,20})\s+:\s*", line.text):
        original = match.group(0)
        if re.search(r"\d\s+:\s*\d", original):
            continue
        suggested = re.sub(r"\s+:\s*", ": ", original)
        findings.append(
            RuleFinding(
                rule_code="COLON_SPACING",
                category="문장부호와 기호",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=line.number,
                line_end=line.number,
                original_text=original,
                suggested_text=suggested,
                reason="시간 표시가 아닌 쌍점(:)은 앞말에 붙이고 뒷말과는 띄어 씁니다.",
            )
        )
    return findings


def _review_time(line: DocumentLine) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    for match in re.finditer(r"\b([01]?\d|2[0-3])\s+:\s*([0-5]\d)\b", line.text):
        hour, minute = match.groups()
        findings.append(
            RuleFinding(
                rule_code="TIME_FORMAT",
                category="시간 표기",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=line.number,
                line_end=line.number,
                original_text=match.group(0),
                suggested_text=f"{int(hour):02d}:{minute}",
                reason="시각은 24시간제 숫자로 쓰고 쌍점 앞뒤를 띄지 않습니다.",
            )
        )
    for match in re.finditer(r"(오전|오후)\s*(\d{1,2})시(?:\s*(\d{1,2})분)?", line.text):
        meridiem, hour_text, minute_text = match.groups()
        hour = int(hour_text)
        minute = int(minute_text or 0)
        if meridiem == "오후" and hour < 12:
            hour += 12
        if meridiem == "오전" and hour == 12:
            hour = 0
        findings.append(
            RuleFinding(
                rule_code="TIME_FORMAT",
                category="시간 표기",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=line.number,
                line_end=line.number,
                original_text=match.group(0),
                suggested_text=f"{hour:02d}:{minute:02d}",
                reason="시각은 오전/오후 표현보다 24시간제 숫자 표기를 우선합니다.",
            )
        )
    return findings


def _review_item_spacing(line: DocumentLine) -> list[RuleFinding]:
    match = ITEM_MARKER_PATTERN.match(line.text)
    if not match:
        return []
    indent = match.group("indent")
    marker = match.group("marker")
    spaces = match.group("spaces")
    content = match.group("content")
    if spaces == " ":
        return []
    return [
        RuleFinding(
            rule_code="ITEM_SPACING",
            category="항목 번호 체계",
            severity="LOW",
            status="REVISION_REQUIRED",
            line_start=line.number,
            line_end=line.number,
            original_text=line.text,
            suggested_text=f"{indent}{marker} {content}",
            reason="항목 기호와 내용 사이에는 한 칸을 띄어 씁니다.",
        )
    ]


def _review_item_marker_hierarchy(lines: list[DocumentLine]) -> list[CheckRequiredItem]:
    body_lines = _body_lines_before_attachments(lines)
    items = [_parse_item_marker(line) for line in body_lines]
    parsed_items = [item for item in items if item is not None]
    if len(parsed_items) < 2:
        return []

    indent_levels = sorted({item.indent_width for item in parsed_items})
    expected_by_indent = {
        indent: ITEM_STYLE_ORDER[min(index, len(ITEM_STYLE_ORDER) - 1)]
        for index, indent in enumerate(indent_levels)
    }

    checks: list[CheckRequiredItem] = []
    reported_indent_levels: set[int] = set()
    for item in parsed_items:
        expected_style = expected_by_indent[item.indent_width]
        expected_indent_width = indent_levels.index(item.indent_width) * 2
        if item.style != expected_style:
            continue
        if item.indent_width != expected_indent_width and item.indent_width not in reported_indent_levels:
            checks.append(
                CheckRequiredItem(
                    category="항목 번호 체계",
                    message=(
                        "항목 들여쓰기 확인이 필요합니다. "
                        f"매뉴얼 기준으로 둘째 항목부터는 위 항목보다 2타씩 오른쪽에서 시작해야 합니다. "
                        f"현재 단계는 {item.indent_width}타 위치로 보이며, 예상 위치는 {expected_indent_width}타입니다."
                    ),
                    original_text=item.line.text.strip(),
                    line_start=item.line.number,
                )
            )
            reported_indent_levels.add(item.indent_width)
    return checks


def _review_item_marker_styles(lines: list[DocumentLine]) -> list[RuleFinding]:
    body_lines = _body_lines_before_attachments(lines)
    items = [_parse_item_marker(line) for line in body_lines]
    parsed_items = [item for item in items if item is not None]
    if len(parsed_items) < 2:
        return []

    indent_levels = sorted({item.indent_width for item in parsed_items})
    expected_by_indent = {
        indent: ITEM_STYLE_ORDER[min(index, len(ITEM_STYLE_ORDER) - 1)]
        for index, indent in enumerate(indent_levels)
    }
    findings: list[RuleFinding] = []
    for index, item in enumerate(parsed_items):
        expected_style = expected_by_indent[item.indent_width]
        sequence_style = _expected_style_from_previous_sequence(parsed_items[:index], item)
        if sequence_style is not None:
            expected_style = sequence_style
        if item.style == expected_style:
            continue
        suggested_line = _replace_item_marker(item.line.text, item.marker, expected_style)
        if suggested_line == item.line.text:
            continue
        findings.append(
            RuleFinding(
                rule_code="ITEM_MARKER_STYLE",
                category="항목 번호 체계",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=item.line.number,
                line_end=item.line.number,
                original_text=item.line.text,
                suggested_text=suggested_line,
                reason=(
                    "항목 단계별 기호는 매뉴얼 순서에 맞춰 "
                    "1., 가., 1), 가), (1), (가), ①, ㉮ 순으로 사용합니다."
                ),
            )
        )
    return findings


def _body_lines_before_attachments(lines: list[DocumentLine]) -> list[DocumentLine]:
    attachment_start = next(
        (
            line.number
            for line in lines
            if "붙임" in line.text or re.search(r"^\s*첨부(?=\s+\d+\.)", line.text)
        ),
        None,
    )
    body_lines = [
        line
        for line in lines
        if attachment_start is None or line.number < attachment_start
    ]
    return body_lines


def _expected_style_from_previous_sequence(
    previous_items: list[ParsedItemMarker],
    item: ParsedItemMarker,
) -> str | None:
    item_value = _marker_value(item.marker)
    item_kind = _marker_kind(item.marker)
    if item_value is None or item_kind is None:
        return None
    for previous in reversed(previous_items):
        previous_value = _marker_value(previous.marker)
        previous_kind = _marker_kind(previous.marker)
        if previous_value is None or previous_kind != item_kind:
            continue
        if previous_value == item_value - 1:
            return previous.style if previous.style.endswith("_dot") else None
        return None
    return None



def _replace_item_marker(line_text: str, current_marker: str, expected_style: str) -> str:
    match = ITEM_MARKER_PATTERN.match(line_text)
    if not match:
        return line_text
    suggested_marker = _convert_marker_to_style(current_marker, expected_style)
    if suggested_marker is None:
        return line_text
    return (
        f"{match.group('indent')}{suggested_marker}"
        f"{match.group('spaces') or ' '}{match.group('content')}"
    )


def _convert_marker_to_style(marker: str, expected_style: str) -> str | None:
    value = _marker_value(marker)
    if value is None:
        return None
    if expected_style == "decimal_dot":
        return f"{value}."
    if expected_style == "korean_dot":
        return f"{_korean_sequence_marker(value)}."
    if expected_style == "decimal_paren":
        return f"{value})"
    if expected_style == "korean_paren":
        return f"{_korean_sequence_marker(value)})"
    if expected_style == "decimal_wrapped":
        return f"({value})"
    if expected_style == "korean_wrapped":
        return f"({_korean_sequence_marker(value)})"
    if expected_style == "circled_digit":
        circled = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
        return circled[value - 1] if 1 <= value <= len(circled) else None
    if expected_style == "circled_korean":
        circled_korean = "㉮㉯㉰㉱㉲㉳㉴㉵㉶㉷㉸㉹㉺㉻"
        return circled_korean[value - 1] if 1 <= value <= len(circled_korean) else None
    return None


def _marker_value(marker: str) -> int | None:
    numeric = re.search(r"\d+", marker)
    if numeric:
        return int(numeric.group(0))
    korean = re.search(r"[가-힣]", marker)
    if korean:
        return _korean_sequence_value(korean.group(0))
    circled = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
    if marker in circled:
        return circled.index(marker) + 1
    circled_korean = "㉮㉯㉰㉱㉲㉳㉴㉵㉶㉷㉸㉹㉺㉻"
    if marker in circled_korean:
        return circled_korean.index(marker) + 1
    return None


def _marker_kind(marker: str) -> str | None:
    if re.search(r"\d+", marker) or marker in "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳":
        return "numeric"
    if re.search(r"[가-힣]", marker) or marker in "㉮㉯㉰㉱㉲㉳㉴㉵㉶㉷㉸㉹㉺㉻":
        return "korean"
    return None


def _korean_sequence_value(char: str) -> int | None:
    order = "가나다라마바사아자차카타파하"
    return order.index(char) + 1 if char in order else None


def _korean_sequence_marker(value: int) -> str:
    order = "가나다라마바사아자차카타파하"
    if 1 <= value <= len(order):
        return order[value - 1]
    return "가"


def _parse_item_marker(line: DocumentLine) -> ParsedItemMarker | None:
    match = ITEM_MARKER_PATTERN.match(line.text)
    if not match:
        return None
    marker = match.group("marker")
    style = _item_marker_style(marker)
    if style is None:
        return None
    return ParsedItemMarker(
        line=line,
        marker=marker,
        style=style,
        indent_width=_indent_width(match.group("indent")),
    )


def _item_marker_style(marker: str) -> str | None:
    if re.fullmatch(r"\d+\.", marker):
        return "decimal_dot"
    if re.fullmatch(r"[가-힣]\.", marker):
        return "korean_dot"
    if re.fullmatch(r"\d+\)", marker):
        return "decimal_paren"
    if re.fullmatch(r"[가-힣]\)", marker):
        return "korean_paren"
    if re.fullmatch(r"\(\d+\)", marker):
        return "decimal_wrapped"
    if re.fullmatch(r"\([가-힣]\)", marker):
        return "korean_wrapped"
    if re.fullmatch(r"[①-⑳]", marker):
        return "circled_digit"
    if re.fullmatch(r"[㉮-㉻]", marker):
        return "circled_korean"
    return None


def _indent_width(indent: str) -> int:
    width = 0
    for char in indent:
        if char == "\t":
            width += 4
        elif char == "\u3000":
            width += 2
        else:
            width += 1
    return width


def _review_approval_phrase(line: DocumentLine) -> list[RuleFinding]:
    approval_request = re.search(r"결재를\s+재가하여 주시기 바랍니다\.?", line.text)
    if approval_request:
        return [
            RuleFinding(
                rule_code="ENDING_EXPRESSION",
                category="문장 종결 표현",
                severity="MEDIUM",
                status="REVISION_REQUIRED",
                line_start=line.number,
                line_end=line.number,
                original_text=approval_request.group(0),
                suggested_text="결재하여 주시기 바랍니다.",
                reason="'재가' 표현은 문서 목적에 따라 결재·보고·협조·참고 표현으로 바꾸어 쓰는 것이 적절합니다.",
            )
        ]
    phrases = ("재가하여 주시기 바랍니다.", "재가하여 주시기 바랍니다")
    phrase = next((candidate for candidate in phrases if candidate in line.text), None)
    if phrase is None:
        return []
    suggested = _suggest_approval_phrase(line.text)
    return [
        RuleFinding(
            rule_code="ENDING_EXPRESSION",
            category="문장 종결 표현",
            severity="MEDIUM",
            status="REVISION_REQUIRED",
            line_start=line.number,
            line_end=line.number,
            original_text=phrase,
            suggested_text=suggested,
            reason="'재가' 표현은 문서 목적에 따라 결재·보고·협조·참고 표현으로 바꾸어 쓰는 것이 적절합니다.",
        )
    ]


def _review_end_marker_spacing(line: DocumentLine) -> list[RuleFinding]:
    if "끝." not in line.text:
        return []
    findings: list[RuleFinding] = []
    for match in re.finditer(r"(?P<prefix>.*?)(?P<spacing>[^\S\r\n]*)끝\.", line.text):
        prefix = match.group("prefix")
        spacing = match.group("spacing")
        stripped_prefix = prefix.rstrip(" \t\u00a0\u3000")
        if stripped_prefix and not stripped_prefix.endswith("."):
            suggested_prefix = f"{stripped_prefix}."
            findings.append(
                RuleFinding(
                    rule_code="END_MARKER",
                    category="끝표시",
                    severity="LOW",
                    status="REVISION_REQUIRED",
                    line_start=line.number,
                    line_end=line.number,
                    original_text=match.group(0),
                    suggested_text=f"{suggested_prefix}  끝.",
                    reason="문장이 마침표 없이 끝난 경우 먼저 마침표를 찍은 뒤 2타를 띄우고 끝표시를 합니다.",
                )
            )
            continue
        if _is_two_end_marker_space_units(spacing):
            continue
        findings.append(
            RuleFinding(
                rule_code="END_MARKER",
                category="끝표시",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=line.number,
                line_end=line.number,
                original_text=match.group(0),
                suggested_text="  끝.",
                reason="끝표시는 본문 또는 붙임 마지막 글자 뒤에 한 글자, 즉 2타를 띄우고 씁니다.",
            )
        )
    return findings


def _is_two_end_marker_space_units(spacing: str) -> bool:
    """Treat HTML non-breaking spaces as visual spaces, but reject tabs."""
    return all(char in {" ", "\u00a0"} for char in spacing) and spacing.replace("\u00a0", " ") == "  "


def _review_attachment_spacing(line: DocumentLine) -> list[RuleFinding]:
    attachment_keyword = re.search(r"^\s*첨부(?=\s+\d+\.)", line.text)
    if attachment_keyword:
        original = line.text
        suggested = re.sub(r"^(\s*)첨부(?=\s+\d+\.)", r"\1붙임", line.text, count=1)
        return [
            RuleFinding(
                rule_code="ATTACHMENT_LABEL",
                category="붙임 표시",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=line.number,
                line_end=line.number,
                original_text=original,
                suggested_text=suggested,
                reason="첨부파일 표시는 '첨부'가 아니라 '붙임'으로 적습니다.",
            )
        ]

    match = re.search(r"붙임\s{0,1}(\d+\.)", line.text)
    if not match:
        return []
    original = match.group(0)
    if original.startswith("붙임  "):
        return []
    return [
        RuleFinding(
            rule_code="ATTACHMENT_SPACING",
            category="붙임 표시",
            severity="LOW",
            status="REVISION_REQUIRED",
            line_start=line.number,
            line_end=line.number,
            original_text=original,
            suggested_text=f"붙임  {match.group(1)}",
            reason="'붙임' 뒤에는 한 글자, 즉 2타를 띄우고 첨부파일명을 적습니다.",
        )
    ]


def _review_amounts(line: DocumentLine) -> list[CheckRequiredItem]:
    checks: list[CheckRequiredItem] = []
    for match in re.finditer(r"금[\d,]+원(?!\()", line.text):
        checks.append(
            CheckRequiredItem(
                category="금액 표기",
                message="금액은 숫자 금액 뒤 괄호 안에 한글 금액을 함께 적는 형식인지 확인해야 합니다.",
                original_text=match.group(0),
                line_start=line.number,
            )
        )
    for match in re.finditer(r"금([\d,]+)원\(금([가-힣]+)원?\)", line.text):
        numeric = int(match.group(1).replace(",", ""))
        korean = _parse_korean_number(match.group(2).removesuffix("원"))
        if korean is None or korean != numeric:
            checks.append(
                CheckRequiredItem(
                    category="금액 표기",
                    message="숫자 금액과 한글 금액의 일치 여부를 확인해야 합니다.",
                    original_text=match.group(0),
                    line_start=line.number,
                )
            )
    return checks


def _review_document_level(lines: list[DocumentLine]) -> list[RuleFinding]:
    text = "\n".join(line.text for line in lines).strip()
    if not text or "끝." in text:
        return []
    last = next((line for line in reversed(lines) if line.text.strip()), lines[-1])
    return [
        RuleFinding(
            rule_code="END_MARKER",
            category="끝표시",
            severity="MEDIUM",
            status="REVISION_REQUIRED",
            line_start=last.number,
            line_end=last.number,
            original_text=last.text,
            suggested_text=_append_end_marker(last.text),
            reason="문서 마지막에는 끝표시가 필요합니다.",
        )
    ]


def _review_single_item_sections(lines: list[DocumentLine]) -> list[RuleFinding]:
    non_empty = [line for line in lines if line.text.strip()]
    top_level = [
        line
        for line in non_empty
        if re.match(r"^\s*\d+\.\s+\S", line.text)
    ]
    if len(top_level) != 1:
        return []
    line = top_level[0]
    return [
        RuleFinding(
            rule_code="SINGLE_ITEM_NUMBERING",
            category="항목 번호 체계",
            severity="LOW",
            status="CHECK_REQUIRED",
            line_start=line.number,
            line_end=line.number,
            original_text=line.text,
            suggested_text=None,
            reason="항목이 하나만 있는 경우 항목기호를 부여하지 않는 것이 원칙입니다. 실제 하위 항목 존재 여부를 확인해 주세요.",
            editable=False,
        )
    ]


def _review_related_documents(lines: list[DocumentLine]) -> list[CheckRequiredItem]:
    checks: list[CheckRequiredItem] = []
    pattern = re.compile(r"[가-힣A-Za-z]+-[0-9]+\(20\d{2}\. \d{1,2}\. \d{1,2}\.,\s*[\"“].+[\"”]\)호")
    for line in lines:
        stripped = line.text.strip()
        if not stripped.startswith("관련"):
            continue
        if pattern.search(stripped):
            continue
        checks.append(
            CheckRequiredItem(
                category="관련문서 표시",
                message="관련문서는 고유문서번호, 날짜, 제목을 포함한 형식인지 확인해야 합니다. 예: 관련: 총무인사팀-123(2025. 4. 3., “출장허가신청서(OOO)”)호",
                original_text=stripped,
                line_start=line.number,
            )
        )
    return checks


def _review_law_references(lines: list[DocumentLine]) -> list[CheckRequiredItem]:
    checks: list[CheckRequiredItem] = []
    for line in lines:
        stripped = line.text.strip()
        if "법" not in stripped or "제" not in stripped or "조" not in stripped:
            continue
        if "「" in stripped and "」" in stripped and re.search(r"제\s*\d+\s*조", stripped):
            continue
        checks.append(
            CheckRequiredItem(
                category="법령 표시",
                message="법령명은 홑낫표(「」)로 묶고 조문 번호와 조문명을 함께 표시했는지 확인해야 합니다.",
                original_text=stripped,
                line_start=line.number,
            )
        )
    return checks


def _review_budget_tables(lines: list[DocumentLine], tables: list[ExtractedTable]) -> list[CheckRequiredItem]:
    text = "\n".join(line.text for line in lines)
    if "소요예산" not in text and "예산" not in text:
        return []
    if re.search(r"별도\s*예산\s*(없이|없음|미사용)|예산\s*(없음|미사용)", text):
        return []

    budget_headers = {"회계연도", "예산구분", "회계구분", "세목", "세목코드", "소요예산"}
    candidate_tables = [
        table
        for table in tables
        if any(header in _flatten_table(table) for header in budget_headers)
    ]
    if not candidate_tables:
        if "소요예산" in text:
            return [
                CheckRequiredItem(
                    category="소요예산 표시",
                    message="예산을 사용하는 기안문이면 소요예산 표가 필요합니다. HTML 표가 인식되지 않아 표 구조 확인이 필요합니다.",
                    original_text="소요예산",
                    line_start=_first_line_number_containing(lines, "소요예산"),
                )
            ]
        return []

    required_labels = {
        "회계연도": ("회계연도", "회계 연도"),
        "회계구분": ("회계구분", "회계 구분", "예산구분", "예산 구분"),
        "세목": ("세목",),
        "세목코드": ("세목코드", "세목 코드"),
        "소요예산": ("소요예산", "소요 예산"),
        "합계": ("합계", "총계"),
    }
    checks: list[CheckRequiredItem] = []
    for table in candidate_tables:
        flattened = _flatten_table(table)
        missing = [
            label
            for label, aliases in required_labels.items()
            if not any(alias in flattened for alias in aliases)
        ]
        if missing:
            checks.append(
                CheckRequiredItem(
                    category="소요예산 표시",
                    message=f"소요예산 표 필수 항목 확인이 필요합니다: {', '.join(missing)}",
                    original_text=f"표 {table.index}",
                )
            )
    return checks


def review_table_checks(text: str, tables: list[ExtractedTable] | None = None) -> list[TableCheckItem]:
    """표는 자동 수정하지 않고, 원본 표에서 사람이 확인할 검토 결과만 만든다."""
    extracted_tables = tables or []
    checks: list[TableCheckItem] = []
    declared_amount = _extract_declared_settlement_amount(text)
    has_detail_context = _has_detail_table_context(text)
    has_budget_context = _has_budget_table_context(text)
    detail_table = _find_detail_amount_table(extracted_tables) if has_detail_context else None
    budget_table = _find_budget_amount_table(extracted_tables) if has_budget_context else None

    if detail_table:
        checks.extend(_review_detail_amount_table(detail_table, declared_amount))
    elif has_detail_context:
        checks.append(
            TableCheckItem(
                table_index=0,
                table_title="수입 정산 상세 내역",
                category="표 검토",
                severity="MEDIUM",
                status="CHECK_REQUIRED",
                message="수입 정산 상세 내역 표 구조가 HTML 표로 인식되지 않았습니다.",
                suggestion="원본 전자결재/HWP 표에서 구분, 건수, 금액, 비고와 합계 행을 직접 확인해 주세요.",
                evidence={"declaredAmount": declared_amount},
            )
        )

    if budget_table:
        checks.extend(_review_budget_amount_table(budget_table, declared_amount, detail_table))
    elif has_budget_context:
        checks.append(
            TableCheckItem(
                table_index=0,
                table_title="소요예산",
                category="표 검토",
                severity="MEDIUM",
                status="CHECK_REQUIRED",
                message="소요예산 표 구조가 HTML 표로 인식되지 않았습니다.",
                suggestion="원본 전자결재/HWP 표에서 회계연도, 예산구분, 세목, 세목코드, 소요예산 항목을 직접 확인해 주세요.",
                evidence={"declaredAmount": declared_amount},
            )
        )

    return checks


def _has_detail_table_context(text: str) -> bool:
    return ("정산" in text or "수입" in text) and ("상세 내역" in text or "상세내역" in text)


def _has_budget_table_context(text: str) -> bool:
    if re.search(r"별도\s*예산\s*(없이|없음|미사용)|예산\s*(없음|미사용)", text):
        return False
    return "소요예산" in text


def _review_detail_amount_table(
    table: ExtractedTable,
    declared_amount: int | None,
) -> list[TableCheckItem]:
    checks: list[TableCheckItem] = []
    flattened = _flatten_table(table)
    required_headers = ("구분", "건수", "금액", "비고")
    missing_headers = [header for header in required_headers if header not in flattened]
    if missing_headers:
        checks.append(
            TableCheckItem(
                table_index=table.index,
                table_title="수입 정산 상세 내역",
                category="표 검토",
                severity="MEDIUM",
                status="CHECK_REQUIRED",
                message=f"상세 내역 표 필수 항목 확인이 필요합니다: {', '.join(missing_headers)}",
                suggestion="원본 표에서 필수 열 제목과 병합 셀 구조가 맞는지 확인해 주세요.",
                evidence={"missingHeaders": missing_headers},
            )
        )

    total_amount = _extract_total_row_amount(table)
    item_amount_sum = _sum_detail_item_amounts(table)
    if item_amount_sum is not None and total_amount is not None and _amounts_differ(item_amount_sum, total_amount):
        diff = abs(item_amount_sum - total_amount)
        checks.append(
            TableCheckItem(
                table_index=table.index,
                table_title="수입 정산 상세 내역",
                category="표 검토",
                severity=_severity_for_amount_diff(diff),
                status="CHECK_REQUIRED",
                message=f"상세 내역 항목 금액 합계 {item_amount_sum:,}원이 합계 행 {total_amount:,}원과 {diff:,}원 차이납니다.",
                suggestion="원본 표에서 각 항목 금액 또는 합계 행 금액을 확인해 주세요.",
                evidence={
                    "itemAmountSum": item_amount_sum,
                    "tableTotalAmount": total_amount,
                    "difference": diff,
                },
            )
        )

    if declared_amount is not None and total_amount is not None and _amounts_differ(declared_amount, total_amount):
        diff = abs(declared_amount - total_amount)
        checks.append(
            TableCheckItem(
                table_index=table.index,
                table_title="수입 정산 상세 내역",
                category="표 검토",
                severity=_severity_for_amount_diff(diff),
                status="CHECK_REQUIRED",
                message=f"본문 정산 금액 {declared_amount:,}원과 상세 내역 합계 {total_amount:,}원이 {diff:,}원 차이납니다.",
                suggestion="본문 정산 금액 또는 원본 표의 합계 금액 중 어떤 값이 맞는지 확인해 주세요.",
                evidence={
                    "declaredAmount": declared_amount,
                    "tableTotalAmount": total_amount,
                    "difference": diff,
                },
            )
        )
    return checks


def _review_budget_amount_table(
    table: ExtractedTable,
    declared_amount: int | None,
    detail_table: ExtractedTable | None,
) -> list[TableCheckItem]:
    checks: list[TableCheckItem] = []
    flattened = _flatten_table(table)
    required_headers = ("회계연도", "세목", "세목코드", "소요예산", "합계")
    missing_headers = [header for header in required_headers if header not in flattened]
    if "예산구분" not in flattened and "회계구분" not in flattened:
        missing_headers.append("예산구분")
    if missing_headers:
        checks.append(
            TableCheckItem(
                table_index=table.index,
                table_title="소요예산",
                category="표 검토",
                severity="MEDIUM",
                status="CHECK_REQUIRED",
                message=f"소요예산 표 필수 항목 확인이 필요합니다: {', '.join(missing_headers)}",
                suggestion="원본 소요예산 표에서 필수 열 제목과 예산 정보를 확인해 주세요.",
                evidence={"missingHeaders": missing_headers},
            )
        )

    budget_amount = _extract_budget_table_amount(table)
    detail_total_amount = _extract_total_row_amount(detail_table) if detail_table else None
    if detail_total_amount is not None and budget_amount is not None and _amounts_differ(detail_total_amount, budget_amount):
        diff = abs(detail_total_amount - budget_amount)
        checks.append(
            TableCheckItem(
                table_index=table.index,
                table_title="소요예산",
                category="표 검토",
                severity=_severity_for_amount_diff(diff),
                status="CHECK_REQUIRED",
                message=f"상세 내역 합계 {detail_total_amount:,}원과 소요예산 {budget_amount:,}원이 {diff:,}원 차이납니다.",
                suggestion="상세 내역 표의 합계와 소요예산 표의 소요예산 금액을 원본에서 대조해 주세요.",
                evidence={
                    "detailTotalAmount": detail_total_amount,
                    "budgetAmount": budget_amount,
                    "difference": diff,
                },
            )
        )
    if declared_amount is not None and budget_amount is not None and _amounts_differ(declared_amount, budget_amount):
        diff = abs(declared_amount - budget_amount)
        checks.append(
            TableCheckItem(
                table_index=table.index,
                table_title="소요예산",
                category="표 검토",
                severity=_severity_for_amount_diff(diff),
                status="CHECK_REQUIRED",
                message=f"본문 정산 금액 {declared_amount:,}원과 소요예산 {budget_amount:,}원이 {diff:,}원 차이납니다.",
                suggestion="본문 정산 금액과 소요예산 표 금액 중 어떤 값이 맞는지 확인해 주세요.",
                evidence={
                    "declaredAmount": declared_amount,
                    "budgetAmount": budget_amount,
                    "difference": diff,
                },
            )
        )
    return checks


def _find_detail_amount_table(tables: list[ExtractedTable]) -> ExtractedTable | None:
    for table in tables:
        flattened = _flatten_table(table)
        if not all(keyword in flattened for keyword in ("구분", "건수", "금액", "비고")):
            continue
        if _extract_total_row_amount(table) is None:
            continue
        if _sum_detail_item_amounts(table) is None:
            continue
        return table
    return None


def _find_budget_amount_table(tables: list[ExtractedTable]) -> ExtractedTable | None:
    for table in tables:
        flattened = _flatten_table(table)
        if all(keyword in flattened for keyword in ("회계연도", "세목", "소요예산")) and _extract_budget_table_amount(table) is not None:
            return table
    return None


def _extract_declared_settlement_amount(text: str) -> int | None:
    # 본문 기준 금액은 명시적으로 "정산 금액"에 붙은 값을 우선 사용한다.
    # 다른 금액(연회비, 단가, 계좌 거래 금액 등)은 표 대조 기준으로 삼지 않는다.
    for pattern in (
        r"정산\s*금액\s*[:：]?\s*금?\s*([0-9,]+)\s*원",
        r"정산\s*금액\s*[:：]?.*?([0-9,]+)\s*원",
    ):
        match = re.search(pattern, text)
        if match:
            return _parse_amount(match.group(1))
    return None


def _extract_total_row_amount(table: ExtractedTable | None) -> int | None:
    if not table:
        return None
    for row in table.rows:
        row_label = _normalize_label_text(" ".join(row))
        if "소계" in row_label or "누계" in row_label:
            continue
        if "합계" not in row_label and "총계" not in row_label:
            continue
        amounts = [_parse_amount(cell) for cell in row]
        values = [amount for amount in amounts if amount is not None]
        if values:
            return values[-1]
    return None


def _sum_detail_item_amounts(table: ExtractedTable) -> int | None:
    amounts: list[int] = []
    for row in table.rows:
        row_text = " ".join(row)
        normalized_row = _normalize_label_text(row_text)
        if any(total_label in normalized_row for total_label in ("합계", "총계", "소계", "누계")):
            continue
        if any(header in normalized_row for header in ("구분", "건수", "금액", "비고")):
            continue
        if len(row) >= 3:
            amount = _parse_amount(row[-2]) or _parse_amount(row[-1])
            if amount is not None:
                amounts.append(amount)
    return sum(amounts) if amounts else None


def _amounts_differ(left: int, right: int) -> bool:
    return abs(left - right) > AMOUNT_TOLERANCE_WON


def _severity_for_amount_diff(diff: int) -> str:
    if diff <= SMALL_AMOUNT_DIFF_WON:
        return "MEDIUM"
    return "HIGH"


def _extract_budget_table_amount(table: ExtractedTable) -> int | None:
    total_amount = _extract_total_row_amount(table)
    if total_amount is not None:
        return total_amount
    for row in table.rows[1:] if len(table.rows) > 1 else table.rows:
        amounts = [_parse_amount(cell) for cell in row]
        values = [amount for amount in amounts if amount is not None]
        if values:
            return values[-1]
    return None


def _parse_amount(value: str | None) -> int | None:
    if not value:
        return None
    compact = re.sub(r"[ \t\u00a0\u2000-\u200b\u202f\u3000]+", "", value)
    matches = list(re.finditer(r"([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*원", compact))
    match = matches[-1] if matches else re.search(r"([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)", compact)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _review_attachment_list(lines: list[DocumentLine]) -> list[CheckRequiredItem]:
    has_attachment_keyword = any("붙임" in line.text or re.search(r"^\s*첨부(?=\s+\d+\.)", line.text) for line in lines)
    if not has_attachment_keyword:
        return []
    first_attachment_index = next(
        (index for index, line in enumerate(lines) if "붙임" in line.text or re.search(r"^\s*첨부(?=\s+\d+\.)", line.text)),
        None,
    )
    attachment_scope = lines[first_attachment_index:] if first_attachment_index is not None else []
    attachment_lines = [
        line
        for line in attachment_scope
        if re.match(r"^\s*(붙임\s+)?\d+\.\s+\S", line.text)
    ]
    checks: list[CheckRequiredItem] = []
    for line in lines:
        if "붙임" not in line.text and not re.search(r"^\s*첨부(?=\s+\d+\.)", line.text):
            continue
        if re.search(r"붙임\s{2}\d+\.\s+.+\s+\d+부\.", line.text):
            continue
        checks.append(
            CheckRequiredItem(
                category="붙임 표시",
                message="'붙임' 뒤 2타 띄움, 첨부파일명, 부수, 마침표 형식인지 확인해야 합니다.",
                original_text=line.text.strip(),
                line_start=line.number,
            )
        )
    if len(attachment_lines) >= 2:
        for line in attachment_lines[1:]:
            if "붙임" in line.text:
                checks.append(
                    CheckRequiredItem(
                        category="붙임 표시",
                        message="두 번째 붙임부터는 '붙임'을 반복하지 않고 첫 번째 첨부파일명 위치에 맞추는 것이 원칙입니다.",
                        original_text=line.text.strip(),
                        line_start=line.number,
                    )
                )
    return checks


def _suggest_approval_phrase(text: str) -> str:
    if any(keyword in text for keyword in ("협조", "요청", "제출", "회신")):
        return "협조하여 주시기 바랍니다."
    if any(keyword in text for keyword in ("보고", "정산", "결과")):
        return "보고합니다."
    if any(keyword in text for keyword in ("안내", "참고")):
        return "업무에 참고하시기 바랍니다."
    return "결재하여 주시기 바랍니다."


def _has_nested_items(lines: list[DocumentLine]) -> bool:
    markers = set()
    for line in lines:
        match = re.match(r"^\s*(\d+\.|[가-힣]\.|\d+\)|[가-힣]\)|\(\d+\)|\([가-힣]\)|[①-⑳]|[㉮-㉻])\s+", line.text)
        if match:
            markers.add(match.group(1))
    return len(markers) >= 2


def _flatten_table(table: ExtractedTable) -> str:
    return " ".join(_normalize_label_text(cell) for row in table.rows for cell in row if cell.strip())


def _normalize_label_text(value: str) -> str:
    return re.sub(r"[ \t\u00a0\u2000-\u200b\u202f\u3000]+", "", value)


def _first_line_number_containing(lines: list[DocumentLine], needle: str) -> int | None:
    for line in lines:
        if needle in line.text:
            return line.number
    return None


def _append_end_marker(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("."):
        return f"{stripped}  끝."
    return f"{stripped}.  끝."


def _parse_korean_number(value: str) -> int | None:
    digits = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
    small_units = {"십": 10, "백": 100, "천": 1000}
    big_units = {"만": 10000, "억": 100000000}
    total = 0
    section = 0
    number = 0
    for char in value.replace("금", ""):
        if char in digits:
            number = digits[char]
        elif char in small_units:
            section += (number or 1) * small_units[char]
            number = 0
        elif char in big_units:
            section += number
            total += (section or 1) * big_units[char]
            section = 0
            number = 0
        else:
            return None
    return total + section + number
