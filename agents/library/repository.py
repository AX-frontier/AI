from __future__ import annotations

from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Engine

from agents.library.db import create_library_engine
from agents.library.models import BookRecord, GuideChunkRecord, GuideDocRecord


class LibraryRepository(Protocol):
    """agent, retriever, test가 공통으로 사용하는 저장소 경계."""

    def search_books(self, keyword: str, limit: int = 5) -> list[BookRecord]:
        ...

    def search_guide_docs(self, keyword: str, limit: int = 3) -> list[GuideDocRecord]:
        ...

    def search_guide_chunks_by_keyword(
        self,
        keyword: str,
        *,
        limit: int = 12,
    ) -> list[GuideChunkRecord]:
        ...

    def search_guide_chunks_by_embedding(
        self,
        query_embedding: list[float],
        *,
        limit: int = 12,
        min_score: float = 0.35,
    ) -> list[GuideChunkRecord]:
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

    def search_guide_chunks_by_keyword(
        self,
        keyword: str,
        *,
        limit: int = 12,
    ) -> list[GuideChunkRecord]:
        """안내 문서 chunk를 제목/본문 키워드 기준으로 검색한다."""
        safe_keyword = _escape_like(keyword)
        if not safe_keyword:
            return []

        query = text(
            """
            SELECT
              chunk.id,
              chunk.guide_doc_id,
              doc.title,
              doc.source_url,
              chunk.content,
              chunk.chunk_index,
              doc.updated_at,
              0.5
                + CASE WHEN doc.title ILIKE :pattern ESCAPE '\\' THEN 0.22 ELSE 0 END
                + CASE WHEN chunk.content ILIKE :pattern ESCAPE '\\' THEN 0.12 ELSE 0 END
                AS score
            FROM library.guide_doc_chunks AS chunk
            JOIN library.guide_docs AS doc ON doc.id = chunk.guide_doc_id
            WHERE doc.title ILIKE :pattern ESCAPE '\\'
               OR chunk.content ILIKE :pattern ESCAPE '\\'
            ORDER BY score DESC, doc.updated_at DESC, chunk.chunk_index ASC
            LIMIT :limit
            """
        )
        with self._engine.connect() as connection:
            rows = connection.execute(
                query,
                {"pattern": f"%{safe_keyword}%", "limit": limit},
            ).mappings()
            return [_guide_chunk_from_row(row) for row in rows]

    def search_guide_chunks_by_embedding(
        self,
        query_embedding: list[float],
        *,
        limit: int = 12,
        min_score: float = 0.35,
    ) -> list[GuideChunkRecord]:
        """pgvector cosine similarity로 의미가 가까운 안내 chunk를 검색한다."""
        embedding_literal = "[" + ",".join(str(value) for value in query_embedding) + "]"
        query = text(
            """
            SELECT
              chunk.id,
              chunk.guide_doc_id,
              doc.title,
              doc.source_url,
              chunk.content,
              chunk.chunk_index,
              doc.updated_at,
              1 - (chunk.embedding <=> CAST(:embedding AS vector)) AS score
            FROM library.guide_doc_chunks AS chunk
            JOIN library.guide_docs AS doc ON doc.id = chunk.guide_doc_id
            WHERE 1 - (chunk.embedding <=> CAST(:embedding AS vector)) >= :min_score
            ORDER BY chunk.embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
            """
        )
        with self._engine.connect() as connection:
            rows = connection.execute(
                query,
                {
                    "embedding": embedding_literal,
                    "limit": limit,
                    "min_score": min_score,
                },
            ).mappings()
            return [_guide_chunk_from_row(row) for row in rows]


def _guide_chunk_from_row(row) -> GuideChunkRecord:
    return GuideChunkRecord(
        id=int(row["id"]),
        guide_doc_id=int(row["guide_doc_id"]),
        title=str(row["title"]),
        source_url=str(row["source_url"]) if row["source_url"] else None,
        content=str(row["content"]),
        chunk_index=int(row["chunk_index"]),
        score=float(row["score"]),
        updated_at=row["updated_at"],
    )


_repository: PostgresLibraryRepository | None = None


def get_library_repository() -> PostgresLibraryRepository:
    """FastAPI 의존성 주입에 사용할 기본 repository를 지연 생성한다."""
    global _repository
    if _repository is None:
        _repository = PostgresLibraryRepository(create_library_engine())
    return _repository
