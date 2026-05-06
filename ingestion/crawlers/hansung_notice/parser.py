from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from .converter import html_to_markdown, is_image_only_content
from .models import Attachment, Notice, NoticeListItem

BASE_URL = "https://www.hansung.ac.kr"
NOTICE_ID_PATTERN = re.compile(r"/bbs/hansung/2127/(\d+)/artclView\.do")


def parse_list(
    html: str,
    source_category_key: str,
    source_category_name: str,
) -> list[NoticeListItem]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[NoticeListItem] = []
    seen_ids: set[str] = set()

    for row in soup.select("table.board-table tbody tr"):
        link = row.select_one("td.td-title a[href*='/bbs/hansung/2127/'][href*='artclView.do']")
        if not link:
            continue

        href = link.get("href", "")
        match = NOTICE_ID_PATTERN.search(href)
        if not match:
            continue

        notice_id = match.group(1)
        if notice_id in seen_ids:
            continue
        seen_ids.add(notice_id)

        title_tag = link.select_one("strong") or link
        category_tag = link.select_one(".cate-name i") or row.select_one(".notice-title")
        cells = row.find_all("td")

        items.append(
            NoticeListItem(
                notice_id=notice_id,
                title=_clean_text(title_tag.get_text(" ", strip=True)),
                category=_clean_text(category_tag.get_text(" ", strip=True))
                if category_tag
                else source_category_name,
                source_category_key=source_category_key,
                source_category_name=source_category_name,
                detail_url=urljoin(BASE_URL, href),
                is_pinned="notice" in row.get("class", []),
                department=_cell_text(cells, "td-write"),
                published_at=_normalize_date(_cell_text(cells, "td-date")),
                views=_to_int(_cell_text(cells, "td-counts")),
            )
        )

    return items


def parse_detail(
    html: str,
    list_item: NoticeListItem,
    *,
    skip_image_only: bool = True,
) -> Notice | None:
    soup = BeautifulSoup(html, "html.parser")
    view = soup.select_one("div.board-view div.view.viewCont")
    if not view:
        raise ValueError("Cannot find notice detail container")

    title = _parse_title(view) or list_item.title
    category = _parse_category(view) or list_item.category
    detail_map = _parse_detail_map(view)
    content = view.select_one("div.txt")
    if not content:
        raise ValueError("Cannot find notice content container")

    content_html = str(content)
    content_markdown = html_to_markdown(content_html)
    if skip_image_only and is_image_only_content(content_html):
        return None

    return Notice(
        notice_id=list_item.notice_id,
        title=title,
        category=category,
        source_category_key=list_item.source_category_key,
        source_category_name=list_item.source_category_name,
        department=detail_map.get("작성자") or list_item.department,
        published_at=_normalize_date(detail_map.get("작성일") or list_item.published_at),
        views=_to_int(detail_map.get("조회수")) or list_item.views,
        source_url=list_item.detail_url,
        content_html=content_html,
        content_markdown=content_markdown,
        attachments=_parse_attachments(view),
    )


def _parse_title(view: Tag) -> str | None:
    title_tag = view.select_one("div.title > strong")
    if not title_tag:
        title_input = view.select_one("#artclViewTitle")
        return title_input.get("value") if title_input else None
    return _clean_text(title_tag.get_text(" ", strip=True))


def _parse_category(view: Tag) -> str | None:
    category_tag = view.select_one("div.title .cate-name i")
    if not category_tag:
        return None
    return _clean_text(category_tag.get_text(" ", strip=True))


def _parse_detail_map(view: Tag) -> dict[str, str]:
    details: dict[str, str] = {}
    for li in view.select("ul.detail li"):
        label = li.select_one("span")
        if not label:
            continue
        key = label.get_text(" ", strip=True).replace(":", "").strip()
        label.extract()
        value = _clean_text(li.get_text(" ", strip=True))
        if key:
            details[key] = value
    return details


def _parse_attachments(view: Tag) -> list[Attachment]:
    attachments: list[Attachment] = []
    for link in view.select("div.attachment a[href*='/download.do']"):
        name_tag = link.select_one("span")
        name = _clean_text(name_tag.get_text(" ", strip=True) if name_tag else link.get_text(" ", strip=True))
        href = link.get("href")
        if name and href:
            attachments.append(Attachment(name=name, url=urljoin(BASE_URL, href)))
    return attachments


def _cell_text(cells: list[Tag], class_name: str) -> str | None:
    for cell in cells:
        if class_name in cell.get("class", []):
            return _clean_text(cell.get_text(" ", strip=True))
    return None


def _normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    match = re.search(r"(\d{4})[.](\d{2})[.](\d{2})", value)
    if not match:
        return value
    return "-".join(match.groups())


def _to_int(value: str | None) -> int | None:
    if not value:
        return None
    digits = re.sub(r"[^\d]", "", value)
    return int(digits) if digits else None


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
