from __future__ import annotations

import json
import os
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Engine

from agents.library.db import create_library_engine
from agents.main_agent.models import MainChunkRecord


class MainChunkRepository(Protocol):
    """Main Agent가 vector DB chunk를 조회할 때 사용하는 저장소 경계."""

    def search_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MainChunkRecord]:
        ...

    def search_keyword_chunks(
        self,
        terms: list[str],
        *,
        limit: int = 10,
    ) -> list[MainChunkRecord]:
        ...


class PostgresMainChunkRepository:
    """pgvector 기반 학교 정보 chunk repository."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def search_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MainChunkRecord]:
        """embedding cosine distance를 similarity score로 바꿔 상위 chunk를 반환한다."""
        embedding_literal = "[" + ",".join(str(value) for value in query_embedding) + "]"
        query = text(
            """
            SELECT
              chunk_id,
              document_id,
              text,
              metadata,
              1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM main_agent.document_chunks
            WHERE 1 - (embedding <=> CAST(:embedding AS vector)) >= :min_score
            ORDER BY embedding <=> CAST(:embedding AS vector)
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
            return [
                MainChunkRecord(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    text=str(row["text"]),
                    score=float(row["score"]),
                    metadata=_normalize_metadata(row["metadata"]),
                )
                for row in rows
            ]

    def search_keyword_chunks(
        self,
        terms: list[str],
        *,
        limit: int = 10,
    ) -> list[MainChunkRecord]:
        strong_terms = [term for term in terms if len(term) >= 2 and term not in {"신청", "기간", "안내"}]
        if not strong_terms:
            return []
        patterns = [f"%{term}%" for term in strong_terms]
        query = text(
            """
            SELECT
              chunk_id,
              document_id,
              text,
              metadata,
              0.5
                + CASE WHEN metadata->>'title' ILIKE ANY(:patterns) THEN 0.18 ELSE 0 END
                + CASE WHEN text ILIKE ANY(:patterns) THEN 0.08 ELSE 0 END
                AS score
            FROM main_agent.document_chunks
            WHERE metadata->>'title' ILIKE ANY(:patterns)
               OR text ILIKE ANY(:patterns)
            ORDER BY score DESC, updated_at DESC
            LIMIT :limit
            """
        )
        with self._engine.connect() as connection:
            rows = connection.execute(query, {"patterns": patterns, "limit": limit}).mappings()
            return [
                MainChunkRecord(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    text=str(row["text"]),
                    score=float(row["score"]),
                    metadata=_normalize_metadata(row["metadata"]),
                )
                for row in rows
            ]


class MockMainChunkRepository:
    """DB 적재 전 로컬 수동 테스트에 쓰는 샘플 chunk repository."""

    def search_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MainChunkRecord]:
        return [
            MainChunkRecord(
                chunk_id="notice-219610-0001",
                document_id="219610",
                text="2026학년도 1학기 복수·부전공 신청 및 변경신청 안내입니다. 신청 대상, 신청 기간, 변경 절차는 공식 학사 공지를 확인해야 합니다.",
                score=0.86,
                metadata={
                    "doc_type": "notice",
                    "notice_id": 219610,
                    "title": "2026학년도 1학기 복수·부전공 신청 및 변경신청 안내",
                    "category": "학사",
                    "department": "학사운영팀",
                    "posted_date": "2026-01-26",
                    "academic_year": 2026,
                    "semester": "1학기",
                    "url": "https://www.hansung.ac.kr/bbs/hansung/2127/219610/artclView.do",
                    "chunk_index": 0,
                },
            )
        ][:limit]

    def search_keyword_chunks(
        self,
        terms: list[str],
        *,
        limit: int = 10,
    ) -> list[MainChunkRecord]:
        return []


def _normalize_metadata(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return {}


_repository: MainChunkRepository | None = None


def get_main_chunk_repository() -> MainChunkRepository:
    """FastAPI 의존성 주입에 사용할 기본 Main Agent repository를 지연 생성한다."""
    global _repository
    if _repository is None:
        if os.getenv("MAIN_AGENT_USE_MOCK_REPOSITORY", "false").lower() == "true":
            _repository = MockMainChunkRepository()
        else:
            _repository = PostgresMainChunkRepository(create_library_engine())
    return _repository
