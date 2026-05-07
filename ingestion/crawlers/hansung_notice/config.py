from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
SOURCE_CONFIG_PATH = ROOT_DIR / "ingestion" / "crawlers" / "hansung-notice" / "sources.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "hansung-notice"


@dataclass(frozen=True, slots=True)
class CrawlerConfig:
    source_config_path: Path = SOURCE_CONFIG_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    list_post_url: str = "https://www.hansung.ac.kr/bbs/hansung/2127/artclList.do"
    request_timeout: float = 15.0
    request_delay_seconds: float = 0.5
    max_pages: int = 1
    since_date: date | None = None
    include_pinned_notices: bool = False
    old_notice_stop_threshold: int = 10
    existing_notice_stop_threshold: int = 5
    skip_image_only: bool = True
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
