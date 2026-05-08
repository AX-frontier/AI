from __future__ import annotations

import os

import pytest

from agents.library.db import create_library_engine
from agents.library.repository import PostgresLibraryRepository
from scripts.evaluate_library_book_search import SEARCH_CASES, evaluate_case, load_book_data_profile


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Actual library.books DB search checks are explicit integration tests.",
)
def test_actual_library_books_search_quality() -> None:
    engine = create_library_engine()
    repository = PostgresLibraryRepository(engine)
    profile = load_book_data_profile(engine)

    assert profile["book_count"] > 0

    results = [evaluate_case(repository, case) for case in SEARCH_CASES]
    failures = [result for result in results if not result.passed]

    assert failures == []
