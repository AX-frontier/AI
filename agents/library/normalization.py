from __future__ import annotations

import html
import re
import unicodedata


def normalize_book_search_text(value: object) -> str:
    """도서 검색용 비교 문자열을 만든다.

    공백/구두점/대소문자/HTML entity 차이를 없애서
    '클린코드'와 '클린 코드', 'Design&amp;IT'와 'design it'을 같은 축에서 검색한다.
    """
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣᄀ-ᇿ]+", "", text)


def build_book_search_text(
    *,
    title: object = None,
    author: object = None,
    publisher: object = None,
    holding_call_no: object = None,
    material_type: object = None,
    location_symbol: object = None,
    stack_location: object = None,
    stack_shelf: object = None,
    isbn: object = None,
) -> str:
    """도서 한 건의 검색 대상 필드를 정규화해 하나의 후보 문자열로 합친다."""
    parts = [
        normalize_book_search_text(value)
        for value in (
            title,
            author,
            publisher,
            holding_call_no,
            material_type,
            location_symbol,
            stack_location,
            stack_shelf,
            isbn,
        )
    ]
    return " ".join(part for part in parts if part)
