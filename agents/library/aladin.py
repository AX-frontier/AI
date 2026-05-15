from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable

import requests

from agents.library.models import BookRecord

LOGGER = logging.getLogger(__name__)
_CACHE: dict[tuple[str, str], tuple[float, AladinPopularity | None]] = {}


@dataclass(frozen=True)
class AladinPopularity:
    """Aladin에서 조회한 외부 인기 신호."""

    popularity_score: float
    matched_by: str


@dataclass(frozen=True)
class AladinBookSignal:
    """Aladin 목록/검색에서 가져온 책 단위 인기 후보."""

    title: str
    isbn: str | None
    isbn13: str | None
    popularity_score: float
    source: str


class AladinClient:
    """library.books 후보를 Aladin 조회 결과로 보강한다.

    외부 API는 추천 품질을 보강하는 선택 신호이므로 실패해도 내부 추천은 계속 진행한다.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 1.5,
        cache_ttl_seconds: int = 60 * 60 * 24,
    ):
        self._api_key = api_key or os.getenv("ALADIN_TTB_KEY")
        self._timeout_seconds = timeout_seconds
        self._cache_ttl_seconds = cache_ttl_seconds

    def enrich(self, books: Iterable[BookRecord]) -> dict[int, AladinPopularity]:
        if not self._api_key:
            return {}

        signals: dict[int, AladinPopularity] = {}
        for book in books:
            signal = self._lookup_book(book)
            if signal is not None:
                signals[book.id] = signal
        return signals

    def bestsellers(self, *, max_results: int = 20) -> list[AladinBookSignal]:
        if not self._api_key:
            return []
        cache_key = ("itemlist", f"bestseller:{max_results}")
        cached = self._load_cached_book_list(cache_key)
        if cached is not None:
            return cached
        try:
            payload = self._request_item_list("Bestseller", max_results=max_results)
        except Exception as exc:
            LOGGER.info("Aladin bestseller lookup failed: %s", exc)
            return []
        items = _extract_book_signals(payload, source="bestseller")
        self._store_cached_book_list(cache_key, items)
        return items

    def _lookup_book(self, book: BookRecord) -> AladinPopularity | None:
        for query, query_type, matched_by in _book_queries(book):
            cached = self._load_cached_signal(query, query_type)
            if cached is not _CACHE_MISS:
                if cached is not None:
                    return cached
                continue
            try:
                payload = self._request(query, query_type)
            except Exception as exc:
                LOGGER.info("Aladin popularity lookup failed: %s", exc)
                return None
            signal = _extract_popularity(payload)
            if signal is not None:
                popularity = AladinPopularity(signal, matched_by)
                self._store_cached_signal(query, query_type, popularity)
                return popularity
            self._store_cached_signal(query, query_type, None)
        return None

    def _load_cached_signal(self, query: str, query_type: str):
        key = _cache_key(query, query_type)
        cached = _CACHE.get(key)
        if cached is None:
            return _CACHE_MISS
        expires_at, signal = cached
        if expires_at < time.time():
            _CACHE.pop(key, None)
            return _CACHE_MISS
        return signal

    def _store_cached_signal(
        self,
        query: str,
        query_type: str,
        signal: AladinPopularity | None,
    ) -> None:
        _CACHE[_cache_key(query, query_type)] = (
            time.time() + self._cache_ttl_seconds,
            signal,
        )

    def _load_cached_book_list(self, key: tuple[str, str]) -> list[AladinBookSignal] | None:
        cached = _BOOK_LIST_CACHE.get(key)
        if cached is None:
            return None
        expires_at, items = cached
        if expires_at < time.time():
            _BOOK_LIST_CACHE.pop(key, None)
            return None
        return items

    def _store_cached_book_list(self, key: tuple[str, str], items: list[AladinBookSignal]) -> None:
        _BOOK_LIST_CACHE[key] = (time.time() + self._cache_ttl_seconds, items)

    def _request(self, query: str, query_type: str) -> dict:
        response = requests.get(
            "https://www.aladin.co.kr/ttb/api/ItemSearch.aspx",
            params={
                "ttbkey": self._api_key,
                "Query": query,
                "QueryType": query_type,
                "SearchTarget": "Book",
                "output": "js",
                "Version": "20131101",
                "MaxResults": 3,
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _request_item_list(self, query_type: str, *, max_results: int) -> dict:
        response = requests.get(
            "https://www.aladin.co.kr/ttb/api/ItemList.aspx",
            params={
                "ttbkey": self._api_key,
                "QueryType": query_type,
                "SearchTarget": "Book",
                "output": "js",
                "Version": "20131101",
                "MaxResults": max_results,
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}


def _book_queries(book: BookRecord) -> list[tuple[str, str, str]]:
    queries: list[tuple[str, str, str]] = []
    if book.isbn:
        isbn = book.isbn.replace("-", "").strip()
        if isbn:
            queries.append((isbn, "ISBN", "isbn"))
    if book.title:
        queries.append((book.title, "Title", "title"))
    return queries


_CACHE_MISS = object()
_BOOK_LIST_CACHE: dict[tuple[str, str], tuple[float, list[AladinBookSignal]]] = {}


def _cache_key(query: str, query_type: str) -> tuple[str, str]:
    return query_type.lower(), query.strip().lower()


def _extract_popularity(payload: dict) -> float | None:
    items = payload.get("item")
    if not isinstance(items, list) or not items:
        return None

    best_score = 0.0
    for item in items:
        if not isinstance(item, dict):
            continue
        sales_point = _as_float(item.get("salesPoint"))
        review_rank = _as_float(item.get("customerReviewRank"))
        best_rank = _as_float(item.get("bestRank"))
        score = min(sales_point / 10000.0, 1.0)
        score += min(review_rank / 10.0, 1.0) * 0.25
        if best_rank > 0:
            score += max(0.0, 1.0 - min(best_rank, 1000.0) / 1000.0) * 0.35
        best_score = max(best_score, score)

    return round(min(best_score, 1.6), 3) if best_score > 0 else None


def _extract_book_signals(payload: dict, *, source: str) -> list[AladinBookSignal]:
    items = payload.get("item")
    if not isinstance(items, list):
        return []
    signals = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        popularity = _extract_popularity({"item": [item]}) or 0.0
        signals.append(
            AladinBookSignal(
                title=title,
                isbn=str(item.get("isbn") or "").strip() or None,
                isbn13=str(item.get("isbn13") or "").strip() or None,
                popularity_score=popularity,
                source=source,
            )
        )
    return signals


def _as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
