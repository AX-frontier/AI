from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class BookRecord:
    id: int
    bib_no: str
    reg_no: str
    title: str
    author: str | None = None
    publisher: str | None = None
    publish_year: int | None = None
    holding_call_no: str | None = None
    material_type: str | None = None
    location_symbol: str | None = None
    stack_location: str | None = None
    stack_shelf: str | None = None
    isbn: str | None = None


@dataclass(frozen=True)
class GuideDocRecord:
    id: int
    source_url: str | None
    title: str
    content: str
    updated_at: datetime | None = None


@dataclass(frozen=True)
class GuideChunkRecord:
    id: int
    guide_doc_id: int
    title: str
    source_url: str | None
    content: str
    chunk_index: int
    score: float
    updated_at: datetime | None = None
