from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Attachment:
    name: str
    url: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class Notice:
    notice_id: str
    title: str
    category: str
    source_category_key: str
    source_category_name: str
    department: str | None
    published_at: str | None
    views: int | None
    source_url: str
    content_html: str
    content_markdown: str
    attachments: list[Attachment] = field(default_factory=list)
    collected_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())

    @property
    def content_path(self) -> str:
        return f"markdown/{self.source_category_key}/{self.notice_id}.md"

    def to_metadata(self) -> dict[str, object]:
        return {
            "notice_id": self.notice_id,
            "source": "hansung_notice",
            "category": self.category,
            "source_category_key": self.source_category_key,
            "source_category_name": self.source_category_name,
            "title": self.title,
            "department": self.department,
            "published_at": self.published_at,
            "views": self.views,
            "source_url": self.source_url,
            "content_path": self.content_path,
            "attachments": [attachment.to_dict() for attachment in self.attachments],
            "collected_at": self.collected_at,
        }

    def to_json_record(self) -> dict[str, object]:
        record = self.to_metadata()
        record["content_html"] = self.content_html
        record["content_markdown"] = self.content_markdown
        return record


@dataclass(frozen=True, slots=True)
class NoticeListItem:
    notice_id: str
    title: str
    category: str
    source_category_key: str
    source_category_name: str
    detail_url: str
    is_pinned: bool = False
    department: str | None = None
    published_at: str | None = None
    views: int | None = None


@dataclass(slots=True)
class CrawlResult:
    saved_count: int = 0
    updated_count: int = 0
    unchanged_count: int = 0
    skip_count: int = 0
    error_count: int = 0
    processed_pages: int = 0
    stopped_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
