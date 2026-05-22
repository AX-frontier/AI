from __future__ import annotations

from bs4 import BeautifulSoup

from agents.document_review.api.schemas import ExtractedTable


def extract_tables_from_html(body_html: str | None) -> list[ExtractedTable]:
    """웹 에디터 HTML에서 table/tr/td 구조를 검토용 JSON으로 변환한다."""
    if not body_html:
        return []

    soup = BeautifulSoup(body_html, "html.parser")
    extracted: list[ExtractedTable] = []
    for table_index, table in enumerate(soup.find_all("table"), start=1):
        rows = _extract_table_grid(table)
        if not rows:
            continue
        extracted.append(
            ExtractedTable(
                index=table_index,
                rowCount=len(rows),
                columnCount=max(len(row) for row in rows),
                rows=rows,
            )
        )
    return extracted


def _extract_table_grid(table) -> list[list[str]]:
    """rowspan/colspan을 반영해 사람이 보는 표의 논리 열 위치를 보존한다."""
    rows: list[list[str]] = []
    rowspans: dict[int, tuple[int, str]] = {}

    for tr in table.find_all("tr"):
        row: list[str] = []
        column = 0
        for cell in tr.find_all(["th", "td"], recursive=False):
            column = _append_pending_rowspans(row, rowspans, column)
            text = cell.get_text(" ", strip=True)
            colspan = _positive_int(cell.get("colspan"), default=1)
            rowspan = _positive_int(cell.get("rowspan"), default=1)

            for offset in range(colspan):
                row.append(text)
                if rowspan > 1:
                    rowspans[column + offset] = (rowspan - 1, text)
            column += colspan

        _append_pending_rowspans(row, rowspans, column)
        if row and any(cell.strip() for cell in row):
            rows.append(row)
    return rows


def _append_pending_rowspans(row: list[str], rowspans: dict[int, tuple[int, str]], column: int) -> int:
    while column in rowspans:
        remaining, text = rowspans[column]
        row.append(text)
        if remaining <= 1:
            del rowspans[column]
        else:
            rowspans[column] = (remaining - 1, text)
        column += 1
    return column


def _positive_int(value: str | None, *, default: int) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
