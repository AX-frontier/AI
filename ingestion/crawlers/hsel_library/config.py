from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "hsel-library"


@dataclass(frozen=True, slots=True)
class HselCrawlerConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    base_url: str = "https://hsel.hansung.ac.kr/"
    notice_list_url: str = "https://hsel.hansung.ac.kr/sb/default_notice_list.mir"
    notice_view_url: str = "https://hsel.hansung.ac.kr/sb/default_notice_view.mir"
    request_timeout: float = 15.0
    request_delay_seconds: float = 0.4
    max_notice_pages: int = 30
    since_date: date | None = None
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

