from __future__ import annotations

from sqlalchemy import text

from agents.library.db import create_library_engine
from agents.library.normalization import build_book_search_text, normalize_book_search_text

BATCH_SIZE = 1000


def main() -> None:
    engine = create_library_engine()
    _ensure_columns(engine)

    total = 0
    last_id = 0
    while True:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                      id,
                      title,
                      author,
                      publisher,
                      holding_call_no,
                      material_type,
                      location_symbol,
                      stack_location,
                      stack_shelf,
                      isbn
                    FROM library.books
                    WHERE id > :last_id
                    ORDER BY id
                    LIMIT :limit
                    """
                ),
                {"last_id": last_id, "limit": BATCH_SIZE},
            ).mappings().all()

        if not rows:
            break

        payload = []
        for row in rows:
            last_id = int(row["id"])
            payload.append(
                {
                    "id": last_id,
                    "title_normalized": normalize_book_search_text(row["title"]),
                    "author_normalized": normalize_book_search_text(row["author"]),
                    "publisher_normalized": normalize_book_search_text(row["publisher"]),
                    "holding_call_no_normalized": normalize_book_search_text(row["holding_call_no"]),
                    "stack_location_normalized": normalize_book_search_text(row["stack_location"]),
                    "stack_shelf_normalized": normalize_book_search_text(row["stack_shelf"]),
                    "search_text_normalized": build_book_search_text(
                        title=row["title"],
                        author=row["author"],
                        publisher=row["publisher"],
                        holding_call_no=row["holding_call_no"],
                        material_type=row["material_type"],
                        location_symbol=row["location_symbol"],
                        stack_location=row["stack_location"],
                        stack_shelf=row["stack_shelf"],
                        isbn=row["isbn"],
                    ),
                }
            )

        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE library.books
                    SET
                      title_normalized = :title_normalized,
                      author_normalized = :author_normalized,
                      publisher_normalized = :publisher_normalized,
                      holding_call_no_normalized = :holding_call_no_normalized,
                      stack_location_normalized = :stack_location_normalized,
                      stack_shelf_normalized = :stack_shelf_normalized,
                      search_text_normalized = :search_text_normalized,
                      updated_at = now()
                    WHERE id = :id
                    """
                ),
                payload,
            )
        total += len(payload)
        print(f"backfilled {total} books")

    _ensure_indexes(engine)
    print(f"done: {total} books")


def _ensure_columns(engine) -> None:
    statements = [
        "ALTER TABLE library.books ADD COLUMN IF NOT EXISTS title_normalized TEXT",
        "ALTER TABLE library.books ADD COLUMN IF NOT EXISTS author_normalized TEXT",
        "ALTER TABLE library.books ADD COLUMN IF NOT EXISTS publisher_normalized TEXT",
        "ALTER TABLE library.books ADD COLUMN IF NOT EXISTS holding_call_no_normalized TEXT",
        "ALTER TABLE library.books ADD COLUMN IF NOT EXISTS stack_location_normalized TEXT",
        "ALTER TABLE library.books ADD COLUMN IF NOT EXISTS stack_shelf_normalized TEXT",
        "ALTER TABLE library.books ADD COLUMN IF NOT EXISTS search_text_normalized TEXT",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _ensure_indexes(engine) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_books_title_normalized_trgm ON library.books USING gin (title_normalized gin_trgm_ops)",
        "CREATE INDEX IF NOT EXISTS idx_books_author_normalized_trgm ON library.books USING gin (author_normalized gin_trgm_ops)",
        "CREATE INDEX IF NOT EXISTS idx_books_publisher_normalized_trgm ON library.books USING gin (publisher_normalized gin_trgm_ops)",
        "CREATE INDEX IF NOT EXISTS idx_books_holding_call_no_normalized_trgm ON library.books USING gin (holding_call_no_normalized gin_trgm_ops)",
        "CREATE INDEX IF NOT EXISTS idx_books_stack_shelf_normalized_trgm ON library.books USING gin (stack_shelf_normalized gin_trgm_ops)",
        "CREATE INDEX IF NOT EXISTS idx_books_search_text_normalized_trgm ON library.books USING gin (search_text_normalized gin_trgm_ops)",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


if __name__ == "__main__":
    main()
