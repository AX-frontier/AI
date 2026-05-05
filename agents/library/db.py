from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


def _normalize_database_url(database_url: str) -> str:
    """일반 PostgreSQL URL을 SQLAlchemy psycopg 드라이버 URL로 변환한다."""
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def get_database_url() -> str:
    """Library Agent가 사용할 DATABASE_URL을 읽고 검증한다."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for Library Agent database access.")
    return _normalize_database_url(database_url)


def create_library_engine() -> Engine:
    """Spring과 공유하는 PostgreSQL DB에 연결할 SQLAlchemy engine을 만든다."""
    return create_engine(get_database_url(), pool_pre_ping=True)
