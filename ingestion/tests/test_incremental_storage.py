from __future__ import annotations

import json

from ingestion.crawlers.hansung_notice.models import Notice
from ingestion.crawlers.hansung_notice.storage import Storage
from ingestion.crawlers.hsel_library.models import HselDocument
from ingestion.crawlers.hsel_library.storage import HselStorage


def test_hansung_notice_storage_detects_unchanged_and_updated_content(tmp_path) -> None:
    storage = Storage(tmp_path)
    notice = _notice("첫 본문")

    assert storage.save_notice(notice) == "created"
    assert storage.save_notice(_notice("첫 본문")) == "unchanged"
    assert storage.save_notice(_notice("수정된 본문")) == "updated"

    record = json.loads((tmp_path / "json" / "academic" / "100.json").read_text())
    assert record["content_markdown"] == "수정된 본문"
    assert record["content_hash"]


def test_hsel_storage_detects_unchanged_and_updated_content(tmp_path) -> None:
    storage = HselStorage(tmp_path)
    document = _hsel_document("운영 시간 안내")

    assert storage.save_document(document) == "created"
    assert storage.save_document(_hsel_document("운영 시간 안내")) == "unchanged"
    assert storage.save_document(_hsel_document("개관 시간 안내")) == "updated"

    record = json.loads((tmp_path / "json" / "pages" / "guide_time.json").read_text())
    assert record["content_markdown"] == "개관 시간 안내"
    assert record["content_hash"]


def _notice(content: str) -> Notice:
    return Notice(
        notice_id="100",
        title="테스트 공지",
        category="학사",
        source_category_key="academic",
        source_category_name="학사공지",
        department="학사팀",
        published_at="2026-05-08",
        views=1,
        source_url="https://example.edu/notices/100",
        content_html=f"<p>{content}</p>",
        content_markdown=content,
        collected_at="2026-05-08T00:00:00+09:00",
    )


def _hsel_document(content: str) -> HselDocument:
    return HselDocument(
        document_id="guide_time",
        doc_type="pages",
        title="개관시간/휴관일 안내",
        source_url="https://hsel.hansung.ac.kr/guide_time.mir",
        content_html=f"<p>{content}</p>",
        content_markdown=content,
        category="도서관 소개",
        collected_at="2026-05-08T00:00:00+09:00",
    )
