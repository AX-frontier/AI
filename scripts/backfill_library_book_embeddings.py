from __future__ import annotations

import argparse
import hashlib
import json
import time

from sqlalchemy import text

from agents.library.db import create_library_engine
from agents.library.normalization import build_book_search_text
from agents.main_agent.embedding import get_embedding_provider


def main() -> None:
    args = _parse_args()
    engine = create_library_engine()
    with engine.begin() as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            text(
                """
                SELECT
                  id,
                  title,
                  author,
                  publisher,
                  publish_year,
                  holding_call_no,
                  material_type,
                  location_symbol,
                  stack_location,
                  stack_shelf,
                  isbn,
                  embedding_text_hash
                FROM library.books
                ORDER BY id
                LIMIT :limit
                """
            ),
            {"limit": args.limit},
        ).mappings().all()

    provider = get_embedding_provider()
    stats = {"updated": 0, "skipped": 0, "processed": 0}
    update_statement = text(
        """
        UPDATE library.books
        SET embedding = CAST(:embedding AS vector),
            embedding_text_hash = :embedding_text_hash,
            updated_at = NOW()
        WHERE id = :id
        """
    )
    started_at = time.monotonic()
    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start : batch_start + args.batch_size]
        pending = []
        for row in batch:
            embedding_text = build_book_embedding_text(row)
            content_hash = _sha256(embedding_text)
            stats["processed"] += 1
            if row["embedding_text_hash"] == content_hash:
                stats["skipped"] += 1
                continue
            pending.append((row, embedding_text, content_hash))

        embeddings = provider.embed_documents([item[1] for item in pending]) if pending else []
        with engine.begin() as connection:
            for (row, _embedding_text, content_hash), embedding in zip(pending, embeddings):
                connection.execute(
                    update_statement,
                    {
                        "id": row["id"],
                        "embedding": _vector_literal(embedding),
                        "embedding_text_hash": content_hash,
                    },
                )
                stats["updated"] += 1
        if not args.quiet:
            elapsed = max(time.monotonic() - started_at, 0.001)
            rate = stats["processed"] / elapsed
            print(
                json.dumps(
                    {
                        "status": "running",
                        "processed": stats["processed"],
                        "total": len(rows),
                        "updated": stats["updated"],
                        "skipped": stats["skipped"],
                        "rate_per_sec": round(rate, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "embedding_provider": provider.__class__.__name__,
                "embedding_dimensions": provider.dimensions,
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_book_embedding_text(row) -> str:
    search_text = build_book_search_text(
        title=row["title"],
        author=row["author"],
        publisher=row["publisher"],
        holding_call_no=row["holding_call_no"],
        material_type=row["material_type"],
        location_symbol=row["location_symbol"],
        stack_location=row["stack_location"],
        stack_shelf=row["stack_shelf"],
        isbn=row["isbn"],
    )
    parts = [
        f"서명: {row['title']}",
        f"저자: {row['author']}" if row["author"] else "",
        f"출판사: {row['publisher']}" if row["publisher"] else "",
        f"출판년도: {row['publish_year']}" if row["publish_year"] else "",
        f"자료유형: {row['material_type']}" if row["material_type"] else "",
        f"청구기호: {row['holding_call_no']}" if row["holding_call_no"] else "",
        f"소장위치: {row['stack_location']} {row['stack_shelf'] or ''}".strip()
        if row["stack_location"]
        else "",
        f"검색텍스트: {search_text}" if search_text else "",
    ]
    return "\n".join(part for part in parts if part).strip()


def _ensure_schema(connection) -> None:
    for statement in (
        "ALTER TABLE library.books ADD COLUMN IF NOT EXISTS embedding vector(1536)",
        "ALTER TABLE library.books ADD COLUMN IF NOT EXISTS embedding_text_hash TEXT",
        """
        CREATE INDEX IF NOT EXISTS idx_books_embedding
        ON library.books USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        WHERE embedding IS NOT NULL
        """,
    ):
        connection.execute(text(statement))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill pgvector embeddings for library.books.")
    parser.add_argument("--limit", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
