from __future__ import annotations

from agents.library.normalization import build_book_search_text, normalize_book_search_text


def test_normalize_book_search_text_removes_spacing_case_punctuation_and_entities() -> None:
    assert normalize_book_search_text("클린 코드") == "클린코드"
    assert normalize_book_search_text(" Clean-Code ") == "cleancode"
    assert normalize_book_search_text("Design&amp;IT정보센터(6F)") == "designit정보센터6f"


def test_build_book_search_text_combines_normalized_book_fields() -> None:
    text = build_book_search_text(
        title="클린 코드",
        author="Robert C. Martin",
        publisher="인사이트",
        holding_call_no="005.133 ㅁ123ㅋ",
        stack_shelf="3-A-2-e",
    )

    assert "클린코드" in text
    assert "robertcmartin" in text
    assert normalize_book_search_text("005.133 ㅁ123ㅋ") in text
    assert "3a2e" in text
