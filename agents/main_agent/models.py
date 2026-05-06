from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class MainChunkRecord:
    """Vector DB에서 조회한 학교 정보 chunk."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any]

    @property
    def title(self) -> str:
        return str(self.metadata.get("title") or self.document_id)

    @property
    def url(self) -> str | None:
        url = self.metadata.get("url")
        return str(url) if url else None

    @property
    def category(self) -> str | None:
        category = self.metadata.get("category")
        return str(category) if category else None

    @property
    def posted_date(self) -> str | None:
        posted = self.metadata.get("posted_date")
        if isinstance(posted, date):
            return posted.isoformat()
        return str(posted) if posted else None
