from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, HselCrawlerConfig
from .service import HselLibraryCrawler


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl Hansung academic library pages/notices.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--since-date", type=date.fromisoformat, default=None)
    parser.add_argument("--max-notice-pages", type=int, default=30)
    parser.add_argument("--pages-only", action="store_true")
    parser.add_argument("--notices-only", action="store_true")
    args = parser.parse_args()

    config = HselCrawlerConfig(
        output_dir=args.output_dir,
        since_date=args.since_date,
        max_notice_pages=args.max_notice_pages,
    )
    result = HselLibraryCrawler(config).run(
        crawl_pages=not args.notices_only,
        crawl_notices=not args.pages_only,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

