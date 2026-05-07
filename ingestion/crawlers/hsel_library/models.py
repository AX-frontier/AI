from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class HselAttachment:
    name: str
    url: str


@dataclass(frozen=True, slots=True)
class HselDocument:
    document_id: str
    doc_type: str
    title: str
    source_url: str
    content_html: str
    content_markdown: str
    category: str | None = None
    published_at: str | None = None
    author: str | None = None
    views: int | None = None
    attachments: list[HselAttachment] = field(default_factory=list)
    collected_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())

    @property
    def content_path(self) -> str:
        return f"markdown/{self.doc_type}/{self.document_id}.md"

    def to_metadata(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "source": "hsel_library",
            "doc_type": self.doc_type,
            "title": self.title,
            "category": self.category,
            "published_at": self.published_at,
            "author": self.author,
            "views": self.views,
            "source_url": self.source_url,
            "content_path": self.content_path,
            "attachments": [asdict(item) for item in self.attachments],
            "collected_at": self.collected_at,
        }

    def to_json_record(self) -> dict[str, object]:
        record = self.to_metadata()
        record["content_html"] = self.content_html
        record["content_markdown"] = self.content_markdown
        return record


@dataclass(frozen=True, slots=True)
class HselNoticeListItem:
    notice_id: str
    title: str
    detail_url: str
    published_at: str | None
    views: int | None


@dataclass(slots=True)
class HselCrawlResult:
    page_count: int = 0
    notice_count: int = 0
    skip_count: int = 0
    error_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

