from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from ingestion.crawlers.hansung_notice.converter import html_to_markdown

from .models import HselDocument, HselNoticeListItem


PUBLIC_PAGE_PREFIXES = (
    "/guide_",
    "/intro_",
    "/lend_guide.mir",
    "/purchase_guide.mir",
    "/copy_guide.mir",
    "/reading_intro.mir",
    "/perusal_reading_room_guide.mir",
)
PUBLIC_BOARD_PATHS = (
    "/sb/default_electronicfair_list.mir",
    "/sb/faq_faq_list.mir",
)
EXCLUDED_PREFIXES = (
    "/sso_",
    "/lend_lend.mir",
    "/purchase_list.mir",
    "/perusal_reading_room_list.mir",
    "/mynotice_list.mir",
    "/knowledge_collection_list.mir",
    "/data_data.mir",
    "/newdata_list.mir",
    "/popularity_lend_list.mir",
)


def discover_public_pages(html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    pages: list[tuple[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        url = urljoin(base_url, href)
        parsed = urlparse(url)
        if parsed.netloc != "hsel.hansung.ac.kr":
            continue
        if not _is_allowed_page(parsed.path):
            continue
        normalized = parsed._replace(fragment="").geturl()
        if normalized in seen:
            continue
        seen.add(normalized)
        title = _clean_text(anchor.get_text(" ", strip=True)) or Path(parsed.path).stem
        pages.append((title, normalized))
    return pages


def parse_info_page(html: str, url: str, fallback_title: str) -> HselDocument | None:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#content_content")
    if content is None:
        return None
    title = _extract_page_title(soup) or fallback_title
    content_html = str(content)
    content_markdown = html_to_markdown(content_html)
    if not content_markdown:
        return None
    return HselDocument(
        document_id=_page_id(url),
        doc_type="pages",
        title=title,
        category=_extract_breadcrumb_category(soup),
        source_url=url,
        content_html=content_html,
        content_markdown=content_markdown,
    )


def parse_notice_list(html: str, base_view_url: str) -> list[HselNoticeListItem]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[HselNoticeListItem] = []
    for row in soup.select("table tbody tr[onclick*='go_view']"):
        onclick = row.get("onclick") or ""
        match = re.search(r"go_view\('(?P<id>\d+)'", onclick)
        if not match:
            continue
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) < 4:
            continue
        notice_id = match.group("id")
        title = _clean_text(cells[1].get_text(" ", strip=True)).replace(" N", "")
        published_at = _parse_date_text(cells[2].get_text(" ", strip=True))
        views = _to_int(cells[3].get_text(" ", strip=True))
        items.append(
            HselNoticeListItem(
                notice_id=notice_id,
                title=title,
                detail_url=f"{base_view_url}?sb_no={notice_id}",
                published_at=published_at,
                views=views,
            )
        )
    return items


def parse_notice_detail(html: str, item: HselNoticeListItem) -> HselDocument | None:
    soup = BeautifulSoup(html, "html.parser")
    header = soup.select_one(".sc_view_header")
    content = soup.select_one(".view_content")
    if content is None:
        return None

    title = item.title
    if header:
        title_node = header.select_one(".title")
        if title_node:
            title = _clean_text(title_node.get_text(" ", strip=True))
    author, published_at = _extract_notice_header_meta(header)
    published_at = _parse_date_text(published_at) or item.published_at

    _absolutize_links(content, item.detail_url)
    content_html = str(content)
    content_markdown = html_to_markdown(content_html)
    if not content_markdown:
        return None

    return HselDocument(
        document_id=item.notice_id,
        doc_type="notices",
        title=title,
        category="공지사항",
        published_at=published_at,
        author=author,
        views=item.views,
        source_url=item.detail_url,
        content_html=content_html,
        content_markdown=content_markdown,
    )


def _is_allowed_page(path: str) -> bool:
    if any(path.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    return path.startswith(PUBLIC_PAGE_PREFIXES) or path in PUBLIC_BOARD_PATHS


def _extract_page_title(soup: BeautifulSoup) -> str | None:
    node = soup.select_one("#content_header h4")
    return _clean_text(node.get_text(" ", strip=True)) if node else None


def _extract_breadcrumb_category(soup: BeautifulSoup) -> str | None:
    items = [
        _clean_text(item.get_text(" ", strip=True))
        for item in soup.select("#content_header .breadcrumb li")
    ]
    items = [item for item in items if item and item != "Home"]
    return " > ".join(items[:-1]) if len(items) > 1 else None


def _extract_notice_header_meta(header: Tag | None) -> tuple[str | None, str | None]:
    if header is None:
        return None, None
    parts = [_clean_text(item.get_text(" ", strip=True)) for item in header.select("ul li")]
    parts = [part for part in parts if part and part != "ㅣ"]
    author = parts[0] if parts else None
    published_at = parts[1] if len(parts) > 1 else None
    return author, published_at


def _absolutize_links(content: Tag, base_url: str) -> None:
    for tag in content.find_all(["a", "img"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if value:
            tag[attr] = urljoin(base_url, value)


def _page_id(url: str) -> str:
    parsed = urlparse(url)
    stem = Path(parsed.path).stem
    if parsed.query:
        query = parse_qs(parsed.query)
        suffix = "_".join(f"{key}-{values[0]}" for key, values in sorted(query.items()) if values)
        return f"{stem}_{_safe_name(suffix)}" if suffix else stem
    return stem


def _parse_date_text(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(20\d{2})[-.](\d{1,2})[-.](\d{1,2})", value)
    if not match:
        return None
    year, month, day = match.groups()
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", value).strip("_")


def _to_int(value: str) -> int | None:
    try:
        return int(value.replace(",", "").strip())
    except (TypeError, ValueError):
        return None

