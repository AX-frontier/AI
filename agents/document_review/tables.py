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
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in tr.find_all(["th", "td"], recursive=False)
            ]
            if cells:
                rows.append(cells)
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
