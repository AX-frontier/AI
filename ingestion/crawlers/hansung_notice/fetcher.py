from __future__ import annotations

import time
from typing import Any

import requests

from .config import CrawlerConfig


class Fetcher:
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )

    def get(self, url: str) -> str:
        response = self.session.get(url, timeout=self.config.request_timeout)
        response.raise_for_status()
        self._delay()
        return response.text

    def post(self, url: str, data: dict[str, Any]) -> str:
        response = self.session.post(url, data=data, timeout=self.config.request_timeout)
        response.raise_for_status()
        self._delay()
        return response.text

    def _delay(self) -> None:
        if self.config.request_delay_seconds > 0:
            time.sleep(self.config.request_delay_seconds)

