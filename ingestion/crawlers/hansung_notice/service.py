from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .config import CrawlerConfig
from .fetcher import Fetcher
from .models import CrawlResult, NoticeListItem
from .parser import parse_detail, parse_list
from .storage import Storage


class HansungNoticeCrawler:
    def __init__(self, config: CrawlerConfig | None = None):
        self.config = config or CrawlerConfig()
        self.fetcher = Fetcher(self.config)
        self.storage = Storage(self.config.output_dir)
        self.sources = self._load_sources(self.config.source_config_path)

    def run(
        self,
        *,
        category_keys: list[str] | None = None,
        max_pages: int | None = None,
    ) -> CrawlResult:
        result = CrawlResult()
        categories = self._select_categories(category_keys)
        page_limit = max_pages or self.config.max_pages
        seen_notice_ids: set[str] = set()

        for category in categories:
            existing_streak = 0
            old_notice_streak = 0
            stop_current_category = False
            for page in range(1, page_limit + 1):
                try:
                    html = self._fetch_list_page(category, page)
                    list_items = parse_list(html, category["key"], category["name"])
                    result.processed_pages += 1
                except Exception as error:
                    result.error_count += 1
                    self.storage.log_error("list", {"category": category, "page": page}, error)
                    continue

                if not list_items:
                    break

                for item in list_items:
                    if item.is_pinned and not self.config.include_pinned_notices:
                        result.skip_count += 1
                        continue

                    if self._is_older_than_cutoff(item.published_at):
                        old_notice_streak += 1
                        if old_notice_streak >= self.config.old_notice_stop_threshold:
                            stop_current_category = True
                            break
                        continue
                    old_notice_streak = 0

                    if item.notice_id in seen_notice_ids:
                        result.skip_count += 1
                        continue
                    seen_notice_ids.add(item.notice_id)

                    status = self._process_detail(item, result)
                    if status == "unchanged":
                        existing_streak += 1
                        if existing_streak >= self.config.existing_notice_stop_threshold:
                            result.stopped_reason = "existing_notice_threshold"
                            stop_current_category = True
                            break
                        continue
                    if status in {"created", "updated"}:
                        existing_streak = 0

                if self._page_is_older_than_cutoff(list_items):
                    break

                if stop_current_category:
                    break

        return result

    def _is_older_than_cutoff(self, published_at: str | None) -> bool:
        if not self.config.since_date or not published_at:
            return False
        parsed = _parse_date(published_at)
        return bool(parsed and parsed < self.config.since_date)

    def _page_is_older_than_cutoff(self, items: list[NoticeListItem]) -> bool:
        if not self.config.since_date or not items:
            return False
        dated_items = [
            item for item in items if not item.is_pinned and _parse_date(item.published_at)
        ]
        return bool(dated_items) and all(
            _parse_date(item.published_at) < self.config.since_date for item in dated_items
        )

    def _process_detail(self, item: NoticeListItem, result: CrawlResult) -> str | None:
        try:
            detail_html = self.fetcher.get(item.detail_url)
        except Exception as error:
            result.error_count += 1
            self.storage.log_error("fetch-detail", {"notice_id": item.notice_id}, error)
            return None

        try:
            notice = parse_detail(detail_html, item, skip_image_only=self.config.skip_image_only)
        except Exception as error:
            result.error_count += 1
            self.storage.log_error("parse-detail", {"notice_id": item.notice_id}, error)
            return None

        if notice is None:
            result.skip_count += 1
            return "skipped"

        try:
            status = self.storage.save_notice(notice)
            if status == "created":
                result.saved_count += 1
            elif status == "updated":
                result.updated_count += 1
            elif status == "unchanged":
                result.unchanged_count += 1
            return status
        except Exception as error:
            result.error_count += 1
            self.storage.log_error("save", {"notice_id": item.notice_id}, error)
            return None

    def _fetch_list_page(self, category: dict[str, Any], page: int) -> str:
        if page == 1 and self.config.since_date is None:
            return self.fetcher.get(category["url"])

        return self.fetcher.post(
            self.config.list_post_url,
            {
                "layout": "68616e73756e674040363137324040666e637431",
                "page": page,
                "findType": "",
                "findWord": "",
                "findClSeq": category.get("find_ccl_seq") or "",
                "findOpnwrd": "",
                "rgsBgndeStr": self.config.since_date.isoformat() if self.config.since_date else "",
                "rgsEnddeStr": "",
            },
        )

    def _select_categories(self, keys: list[str] | None) -> list[dict[str, Any]]:
        categories = self.sources["categories"]
        if not keys:
            return categories
        selected = [category for category in categories if category["key"] in keys]
        missing = sorted(set(keys) - {category["key"] for category in selected})
        if missing:
            raise ValueError(f"Unknown category keys: {', '.join(missing)}")
        return selected

    @staticmethod
    def _load_sources(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
