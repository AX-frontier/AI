from __future__ import annotations

from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Engine

from agents.library.db import create_library_engine
from agents.library.models import BookRecord, GuideDocRecord


class LibraryRepository(Protocol):
    """agent, retriever, test가 공통으로 사용하는 저장소 경계."""

    def search_books(self, keyword: str, limit: int = 5) -> list[BookRecord]:
        ...

    def search_guide_docs(self, keyword: str, limit: int = 3) -> list[GuideDocRecord]:
        ...


def _escape_like(value: str) -> str:
    """사용자 입력을 ILIKE 패턴에 넣기 전에 wildcard 문자를 escape한다."""
    return (
        value.strip()
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


class PostgresLibraryRepository:
    """Library Agent 소유 테이블만 읽는 PostgreSQL repository 구현체."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def search_books(self, keyword: str, limit: int = 5) -> list[BookRecord]:
        """도서 메타데이터와 위치 필드를 대상으로 library.books를 검색한다."""
        safe_keyword = _escape_like(keyword)
        if not safe_keyword:
            return []

        query = text(
            """
            SELECT
              id,
              bib_no,
              reg_no,
              title,
              author,
              publisher,
              publish_year,
              holding_call_no,
              material_type,
              location_symbol,
              stack_location,
              stack_shelf
            FROM library.books
            WHERE title ILIKE :pattern ESCAPE '\\'
               OR author ILIKE :pattern ESCAPE '\\'
               OR publisher ILIKE :pattern ESCAPE '\\'
               OR holding_call_no ILIKE :pattern ESCAPE '\\'
               OR stack_location ILIKE :pattern ESCAPE '\\'
               OR stack_shelf ILIKE :pattern ESCAPE '\\'
            ORDER BY title ASC, id ASC
            LIMIT :limit
            """
        )
        with self._engine.connect() as connection:
            rows = connection.execute(
                query,
                {"pattern": f"%{safe_keyword}%", "limit": limit},
            ).mappings()
            return [BookRecord(**row) for row in rows]

    def search_guide_docs(self, keyword: str, limit: int = 3) -> list[GuideDocRecord]:
        """제목과 본문을 대상으로 library.guide_docs를 검색한다."""
        safe_keyword = _escape_like(keyword)
        if not safe_keyword:
            return []

        query = text(
            """
            SELECT id, source_url, title, content, updated_at
            FROM library.guide_docs
            WHERE title ILIKE :pattern ESCAPE '\\'
               OR content ILIKE :pattern ESCAPE '\\'
            ORDER BY updated_at DESC, id ASC
            LIMIT :limit
            """
        )
        with self._engine.connect() as connection:
            rows = connection.execute(
                query,
                {"pattern": f"%{safe_keyword}%", "limit": limit},
            ).mappings()
            return [GuideDocRecord(**row) for row in rows]


_repository: PostgresLibraryRepository | None = None


def get_library_repository() -> PostgresLibraryRepository:
    """FastAPI 의존성 주입에 사용할 기본 repository를 지연 생성한다."""
    global _repository
    if _repository is None:
        _repository = PostgresLibraryRepository(create_library_engine())
    return _repository
