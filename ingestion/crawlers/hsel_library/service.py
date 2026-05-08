from __future__ import annotations

from datetime import date

from .config import HselCrawlerConfig
from .fetcher import HselFetcher
from .models import HselCrawlResult, HselNoticeListItem
from .parser import (
    discover_public_pages,
    parse_info_page,
    parse_notice_detail,
    parse_notice_list,
)
from .storage import HselStorage


class HselLibraryCrawler:
    def __init__(self, config: HselCrawlerConfig | None = None):
        self.config = config or HselCrawlerConfig()
        self.fetcher = HselFetcher(self.config)
        self.storage = HselStorage(self.config.output_dir)

    def run(self, *, crawl_pages: bool = True, crawl_notices: bool = True) -> HselCrawlResult:
        result = HselCrawlResult()
        if crawl_pages:
            self._crawl_info_pages(result)
        if crawl_notices:
            self._crawl_notices(result)
        return result

    def _crawl_info_pages(self, result: HselCrawlResult) -> None:
        html = self.fetcher.get(self.config.base_url)
        for title, url in discover_public_pages(html, self.config.base_url):
            try:
                page_html = self.fetcher.get(url)
                document = parse_info_page(page_html, url, title)
                if document is None:
                    result.skip_count += 1
                    continue
                status = self.storage.save_document(document)
                if status == "created":
                    result.page_count += 1
                elif status == "updated":
                    result.updated_count += 1
                elif status == "unchanged":
                    result.unchanged_count += 1
            except Exception as error:
                result.error_count += 1
                self.storage.log_error("page", {"title": title, "url": url}, error)

    def _crawl_notices(self, result: HselCrawlResult) -> None:
        for page in range(1, self.config.max_notice_pages + 1):
            try:
                html = self.fetcher.get(self.config.notice_list_url, params={"page_num": page})
                items = parse_notice_list(html, self.config.notice_view_url)
            except Exception as error:
                result.error_count += 1
                self.storage.log_error("notice-list", {"page": page}, error)
                continue
            if not items:
                break

            if self._page_is_older_than_cutoff(items):
                break

            for item in items:
                if self._is_older_than_cutoff(item.published_at):
                    result.skip_count += 1
                    continue
                self._process_notice(item, result)

    def _process_notice(self, item: HselNoticeListItem, result: HselCrawlResult) -> None:
        try:
            html = self.fetcher.get(item.detail_url)
            document = parse_notice_detail(html, item)
            if document is None:
                result.skip_count += 1
                return
            status = self.storage.save_document(document)
            if status == "created":
                result.notice_count += 1
            elif status == "updated":
                result.updated_count += 1
            elif status == "unchanged":
                result.unchanged_count += 1
        except Exception as error:
            result.error_count += 1
            self.storage.log_error("notice-detail", {"notice_id": item.notice_id}, error)

    def _is_older_than_cutoff(self, published_at: str | None) -> bool:
        if not self.config.since_date or not published_at:
            return False
        return date.fromisoformat(published_at) < self.config.since_date

    def _page_is_older_than_cutoff(self, items: list[HselNoticeListItem]) -> bool:
        if not self.config.since_date:
            return False
        dated = [item for item in items if item.published_at]
        return bool(dated) and all(
            date.fromisoformat(item.published_at) < self.config.since_date for item in dated
        )
