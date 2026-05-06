from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import markdownify as md


def html_to_markdown(content_html: str) -> str:
    soup = BeautifulSoup(content_html, "html.parser")
    _remove_unwanted_nodes(soup)
    _normalize_tables(soup)

    markdown = md(
        str(soup),
        heading_style="ATX",
        bullets="-",
        strip=["script", "style"],
    )
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def is_image_only_content(content_html: str) -> bool:
    soup = BeautifulSoup(content_html, "html.parser")
    has_image = soup.find("img") is not None
    text = soup.get_text(" ", strip=True)
    return has_image and not text


def _remove_unwanted_nodes(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    for tag in soup.find_all(True):
        tag.attrs = {
            key: value
            for key, value in tag.attrs.items()
            if key in {"href", "src", "alt", "title", "colspan", "rowspan"}
        }

    for empty in soup.find_all(lambda tag: tag.name in {"span", "p", "div"} and not tag.get_text(strip=True) and not tag.find("img")):
        empty.decompose()


def _normalize_tables(soup: BeautifulSoup) -> None:
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        cells = table.find_all(["td", "th"])

        if len(cells) == 1:
            wrapper = soup.new_tag("div")
            for child in list(cells[0].contents):
                wrapper.append(child)
            table.replace_with(wrapper)
            continue

        if rows and table.find("th") is None:
            for td in rows[0].find_all("td"):
                td.name = "th"
