from __future__ import annotations

import re
from dataclasses import dataclass

from agents.library.aladin import AladinClient, AladinPopularity
from agents.library.models import BookRecord


@dataclass(frozen=True)
class RankedBook:
    book: BookRecord
    score: float
    reasons: tuple[str, ...]
    basis: tuple[str, ...]
    holding_count: int = 1
    dedupe_key: str | None = None


def rank_book_recommendations(
    keyword: str,
    books: list[BookRecord],
    *,
    aladin_client: AladinClient | None = None,
    aladin_weight: float = 1.0,
) -> list[RankedBook]:
    """내부 소장 도서 후보를 추천용 점수로 재정렬한다."""
    if not books:
        return []

    grouped_books = _dedupe_recommendation_books(books)

    try:
        aladin_signals = (aladin_client or AladinClient()).enrich(
            [group.book for group in grouped_books]
        )
    except Exception:
        aladin_signals = {}
    ranked = [
        _rank_book(
            keyword,
            group.book,
            aladin_signals.get(group.book.id),
            aladin_weight=aladin_weight,
            holding_count=group.holding_count,
            dedupe_key=group.dedupe_key,
        )
        for group in grouped_books
    ]
    return sorted(ranked, key=lambda item: (-item.score, item.book.title, item.book.id))


def build_recommendation_summary(
    ranked_books: list[RankedBook],
    *,
    expandedFrom: str | None = None,
    includeSemantic: bool = False,
    queryInterpretation: dict | None = None,
) -> dict:
    basis = []
    rank_reasons = []
    for ranked in ranked_books:
        for item in ranked.basis:
            if item not in basis:
                basis.append(item)
        rank_reasons.append(
            {
                "bookId": ranked.book.id,
                "title": ranked.book.title,
                "score": round(ranked.score, 3),
                "rankReason": ", ".join(ranked.reasons),
                "holdingCount": ranked.holding_count,
                "dedupeKey": ranked.dedupe_key,
            }
        )
    if "internal_search" not in basis:
        basis.insert(0, "internal_search")
    if includeSemantic and "semantic" not in basis:
        basis.insert(1, "semantic")

    summary = {
        "contentType": "book_recommendation",
        "recommendationBasis": basis,
        "rankReasons": rank_reasons,
    }
    if expandedFrom:
        summary["expandedFrom"] = expandedFrom
    if queryInterpretation:
        summary["queryInterpretation"] = queryInterpretation
    return summary


@dataclass(frozen=True)
class _BookGroup:
    book: BookRecord
    holding_count: int
    dedupe_key: str


def _rank_book(
    keyword: str,
    book: BookRecord,
    aladin: AladinPopularity | None,
    *,
    aladin_weight: float,
    holding_count: int,
    dedupe_key: str,
) -> RankedBook:
    terms = _terms(keyword)
    score = 0.0
    reasons: list[str] = []
    basis = ["internal_search"]

    lexical = _lexical_score(terms, book)
    if lexical > 0:
        score += lexical
        reasons.append("검색어와 도서 메타데이터가 잘 맞음")

    recency = _recency_score(book.publish_year)
    if recency > 0:
        score += recency
        reasons.append("비교적 최근 출간")

    availability = _availability_score(book)
    if availability > 0:
        score += availability
        reasons.append("소장 위치 정보가 충분함")

    if aladin is not None:
        score += aladin.popularity_score * max(aladin_weight, 0.0)
        basis.append("aladin_popularity")
        if aladin_weight > 1.0:
            reasons.append("알라딘 인기 신호를 우선 반영")
        else:
            reasons.append("알라딘 인기 신호 확인")

    if holding_count > 1:
        reasons.append(f"동일 도서 소장 {holding_count}권")

    if not reasons:
        reasons.append("내부 소장 도서 후보")
    return RankedBook(
        book=book,
        score=round(score, 3),
        reasons=tuple(reasons),
        basis=tuple(basis),
        holding_count=holding_count,
        dedupe_key=dedupe_key,
    )


def _dedupe_recommendation_books(books: list[BookRecord]) -> list[_BookGroup]:
    groups: dict[str, list[BookRecord]] = {}
    order: list[str] = []
    for book in books:
        key = _dedupe_key(book)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(book)
    return [
        _BookGroup(
            book=_select_representative_book(groups[key]),
            holding_count=len(groups[key]),
            dedupe_key=key,
        )
        for key in order
    ]


def _select_representative_book(books: list[BookRecord]) -> BookRecord:
    return sorted(
        books,
        key=lambda book: (
            -_availability_score(book),
            -(book.publish_year or 0),
            0 if book.isbn else 1,
            book.id,
        ),
    )[0]


def _dedupe_key(book: BookRecord) -> str:
    parts = [
        _normalize_key_part(book.title),
        _normalize_key_part(book.author),
        _normalize_key_part(book.publisher),
    ]
    return "|".join(parts)


def _normalize_key_part(value: str | None) -> str:
    normalized = re.sub(r"\s+", "", (value or "").lower())
    normalized = re.sub(r"[\[\]().,:;·ㆍ/\\\\-]+", "", normalized)
    return normalized


def _lexical_score(terms: list[str], book: BookRecord) -> float:
    if not terms:
        return 0.4
    fields = {
        "title": (book.title or "").lower(),
        "author": (book.author or "").lower(),
        "publisher": (book.publisher or "").lower(),
        "call_no": (book.holding_call_no or "").lower(),
    }
    score = 0.0
    for term in terms:
        if term in fields["title"]:
            score += 1.0
        if term in fields["author"]:
            score += 0.6
        if term in fields["publisher"]:
            score += 0.35
        if term in fields["call_no"]:
            score += 0.25
    return min(score, 2.2)


def _recency_score(year: int | None) -> float:
    if year is None:
        return 0.0
    if year >= 2024:
        return 0.45
    if year >= 2020:
        return 0.3
    if year >= 2015:
        return 0.15
    return 0.0


def _availability_score(book: BookRecord) -> float:
    score = 0.0
    if book.holding_call_no:
        score += 0.2
    if book.stack_location:
        score += 0.2
    if book.stack_shelf:
        score += 0.15
    return score


def _terms(keyword: str) -> list[str]:
    seen: set[str] = set()
    terms = []
    for term in re.findall(r"[0-9A-Za-z가-힣]+", keyword.lower()):
        if len(term) < 2 or term in {"책", "도서", "추천"} or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms
