from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from ingestion.crawlers.hsel_library.config import DEFAULT_OUTPUT_DIR, HselCrawlerConfig
from ingestion.crawlers.hsel_library.fetcher import HselFetcher
from ingestion.crawlers.hsel_library.parser import parse_info_page
from ingestion.crawlers.hsel_library.storage import HselStorage

DEFAULT_URLS = (
    "https://hsel.hansung.ac.kr/intro_staff.mir",
    "https://hsel.hansung.ac.kr/intro_data.mir",
)


def main() -> None:
    args = _parse_args()
    config = HselCrawlerConfig(output_dir=args.output_dir, request_delay_seconds=args.request_delay_seconds)
    fetcher = HselFetcher(config)
    storage = HselStorage(args.output_dir)
    results = []

    for url in args.urls:
        fallback_title = _fallback_title(url)
        try:
            html = fetcher.get(url)
            document = parse_info_page(html, url, fallback_title)
            if document is None:
                results.append({"url": url, "status": "skipped", "reason": "no parsable content"})
                continue
            status = storage.save_document(document)
            results.append(
                {
                    "url": url,
                    "status": status,
                    "document_id": document.document_id,
                    "title": document.title,
                    "markdown_path": str(args.output_dir / document.content_path),
                }
            )
        except Exception as exc:
            storage.log_error("manual-page", {"url": url}, exc)
            results.append({"url": url, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    print(json.dumps({"output_dir": str(args.output_dir), "results": results}, ensure_ascii=False, indent=2))


def _fallback_title(url: str) -> str:
    stem = Path(urlparse(url).path).stem
    titles = {
        "intro_staff": "조직안내",
        "intro_data": "규정/세칙",
    }
    return titles.get(stem, stem)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl selected HSEL library info pages by URL.")
    parser.add_argument("urls", nargs="*", default=list(DEFAULT_URLS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--request-delay-seconds", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
