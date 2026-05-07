from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, CrawlerConfig
from .service import HansungNoticeCrawler


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl Hansung University notices.")
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--since-date", type=date.fromisoformat, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-image-only", action="store_true")
    parser.add_argument("--include-pinned", action="store_true")
    args = parser.parse_args()

    config = CrawlerConfig(
        max_pages=args.max_pages,
        since_date=args.since_date,
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
        skip_image_only=not args.include_image_only,
        include_pinned_notices=args.include_pinned,
    )
    result = HansungNoticeCrawler(config).run(
        category_keys=args.categories,
        max_pages=args.max_pages,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
