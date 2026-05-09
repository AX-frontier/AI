from __future__ import annotations

import re

from agents.document_review.api.schemas import ExtractedTable
from agents.document_review.models import (
    CheckRequiredItem,
    DocumentLine,
    FormatNoticeItem,
    RuleFinding,
)


def apply_safe_suggestions_to_html(body_html: str | None, findings: list[RuleFinding]) -> str | None:
    if not body_html:
        return None
    revised = body_html
    for finding in findings:
        if not finding.editable or not finding.suggested_text:
            continue
        revised = revised.replace(finding.original_text, finding.suggested_text, 1)
    return revised


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
    checks.extend(_review_related_documents(lines))
    checks.extend(_review_law_references(lines))
    checks.extend(_review_budget_tables(lines, tables))
    checks.extend(_review_attachment_list(lines))
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
    match = re.match(r"^(\s*)(\d+\.|[가-힣]\.|\d+\)|[가-힣]\)|\(\d+\)|\([가-힣]\)|[①-⑳]|[㉮-㉻])(\s*)(\S.*)$", line.text)
    if not match:
        return []
    indent, marker, spaces, content = match.groups()
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
    for match in re.finditer(r"(?<!  )끝\.", line.text):
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


def _review_attachment_spacing(line: DocumentLine) -> list[RuleFinding]:
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


def _review_attachment_list(lines: list[DocumentLine]) -> list[CheckRequiredItem]:
    has_attachment_keyword = any("붙임" in line.text for line in lines)
    if not has_attachment_keyword:
        return []
    first_attachment_index = next(
        (index for index, line in enumerate(lines) if "붙임" in line.text),
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
        if "붙임" not in line.text:
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
    return " ".join(cell.strip() for row in table.rows for cell in row if cell.strip())


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
