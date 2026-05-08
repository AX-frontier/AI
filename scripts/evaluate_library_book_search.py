from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import Callable

from sqlalchemy import text

from agents.library.db import create_library_engine
from agents.library.models import BookRecord
from agents.library.repository import PostgresLibraryRepository


@dataclass(frozen=True)
class BookSearchCase:
    name: str
    keyword: str
    expected_fields: tuple[str, ...]
    require_location_fields: bool = False


@dataclass(frozen=True)
class BookSearchCaseResult:
    name: str
    keyword: str
    passed: bool
    result_count: int
    matched_expected_field: bool
    location_fields_present: bool
    first_result: dict | None
    failure_reason: str | None = None


SEARCH_CASES: tuple[BookSearchCase, ...] = (
    BookSearchCase("title_search", "파이썬", ("title",)),
    BookSearchCase("author_search", "슈카친구들", ("author",)),
    BookSearchCase("publisher_search", "Facet", ("publisher",)),
    BookSearchCase("call_no_search", "001 ㄱ785ㄷ", ("holding_call_no",), True),
    BookSearchCase("stack_location_search", "인문자연과학자료실", ("stack_location",), True),
    BookSearchCase("stack_shelf_search", "1-A-1-d", ("stack_shelf",), True),
)


def main() -> int:
    engine = create_library_engine()
    repository = PostgresLibraryRepository(engine)
    profile = load_book_data_profile(engine)
    case_results = [evaluate_case(repository, case) for case in SEARCH_CASES]
    passed = all(result.passed for result in case_results)
    payload = {
        "passed": passed,
        "profile": profile,
        "cases": [asdict(result) for result in case_results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def load_book_data_profile(engine) -> dict:
    query = text(
        """
        SELECT
          count(*) AS book_count,
          count(*) FILTER (WHERE title IS NOT NULL) AS title_count,
          count(*) FILTER (WHERE author IS NOT NULL) AS author_count,
          count(*) FILTER (WHERE publisher IS NOT NULL) AS publisher_count,
          count(*) FILTER (WHERE holding_call_no IS NOT NULL) AS call_no_count,
          count(*) FILTER (WHERE stack_location IS NOT NULL) AS stack_location_count,
          count(*) FILTER (WHERE stack_shelf IS NOT NULL) AS stack_shelf_count
        FROM library.books
        """
    )
    with engine.connect() as connection:
        row = connection.execute(query).mappings().one()
        return {key: int(value) for key, value in row.items()}


def evaluate_case(
    repository: PostgresLibraryRepository,
    case: BookSearchCase,
    *,
    limit: int = 5,
) -> BookSearchCaseResult:
    books = repository.search_books(case.keyword, limit=limit)
    matched_expected_field = any(
        _book_matches_any_field(book, case.keyword, case.expected_fields) for book in books
    )
    location_fields_present = (
        bool(books)
        and all(
            book.holding_call_no and book.stack_location and book.stack_shelf
            for book in books
        )
        if case.require_location_fields
        else True
    )
    failure_reason = _failure_reason(
        books=books,
        matched_expected_field=matched_expected_field,
        location_fields_present=location_fields_present,
    )
    return BookSearchCaseResult(
        name=case.name,
        keyword=case.keyword,
        passed=failure_reason is None,
        result_count=len(books),
        matched_expected_field=matched_expected_field,
        location_fields_present=location_fields_present,
        first_result=_book_to_dict(books[0]) if books else None,
        failure_reason=failure_reason,
    )


def _book_matches_any_field(
    book: BookRecord,
    keyword: str,
    field_names: tuple[str, ...],
) -> bool:
    normalized_keyword = _normalize(keyword)
    return any(normalized_keyword in _normalize(getattr(book, field_name)) for field_name in field_names)


def _failure_reason(
    *,
    books: list[BookRecord],
    matched_expected_field: bool,
    location_fields_present: bool,
) -> str | None:
    if not books:
        return "no_results"
    if not matched_expected_field:
        return "expected_field_not_matched"
    if not location_fields_present:
        return "missing_location_fields"
    return None


def _book_to_dict(book: BookRecord) -> dict:
    return {
        "id": book.id,
        "title": book.title,
        "author": book.author,
        "publisher": book.publisher,
        "holding_call_no": book.holding_call_no,
        "stack_location": book.stack_location,
        "stack_shelf": book.stack_shelf,
    }


def _normalize(value: object) -> str:
    return str(value or "").lower().replace(" ", "")


if __name__ == "__main__":
    sys.exit(main())
