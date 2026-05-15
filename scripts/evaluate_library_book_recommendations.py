from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from uuid import uuid4

from agents.library.agent import run_library_agent
from agents.library.api.schemas import LibraryChatRequest
from agents.library.repository import get_library_repository


@dataclass(frozen=True)
class RecommendationCase:
    name: str
    message: str
    expected_terms: tuple[str, ...]
    excluded_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecommendationCaseResult:
    name: str
    message: str
    passed: bool
    intent: str
    fallback_used: bool
    result_count: int
    recommendation_basis: list[str]
    duplicate_free: bool
    expected_term_found: bool
    excluded_term_absent: bool
    rank_reasons_present: bool
    top_titles: list[str]
    failure_reasons: list[str]


RECOMMENDATION_CASES: tuple[RecommendationCase, ...] = (
    RecommendationCase("python_intro", "파이썬 입문 책 추천해줘", ("파이썬", "python")),
    RecommendationCase("ai", "인공지능 책 추천해줘", ("인공지능", "ai", "머신러닝", "딥러닝")),
    RecommendationCase("data_analysis", "데이터 분석 책 추천해줘", ("데이터", "분석", "통계")),
    RecommendationCase("economics", "경제 책 추천해줘", ("경제", "금융", "경영")),
    RecommendationCase("novel", "소설 책 추천해줘", ("소설", "문학")),
    RecommendationCase("computer_network", "컴퓨터 네트워크 책 추천해줘", ("네트워크", "컴퓨터")),
    RecommendationCase("database", "데이터베이스 책 추천해줘", ("데이터베이스", "database", "sql")),
    RecommendationCase("design", "디자인 책 추천해줘", ("디자인", "design")),
    RecommendationCase("marketing", "마케팅 책 추천해줘", ("마케팅", "marketing")),
    RecommendationCase("statistics", "통계 책 추천해줘", ("통계", "statistics", "데이터")),
)


def main() -> int:
    args = _parse_args()
    repository = get_library_repository()
    cases = RECOMMENDATION_CASES[: args.limit] if args.limit else RECOMMENDATION_CASES
    results = [evaluate_case(repository, case) for case in cases]
    payload = {
        "passed": all(result.passed for result in results),
        "case_count": len(results),
        "passed_count": sum(1 for result in results if result.passed),
        "cases": [asdict(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


def evaluate_case(repository, case: RecommendationCase) -> RecommendationCaseResult:
    response = run_library_agent(
        LibraryChatRequest(
            queryUid=uuid4(),
            traceId=uuid4(),
            conversationUid=uuid4(),
            message=case.message,
        ),
        repository=repository,
    )
    top_books = response.matchedBooks[:5]
    top_texts = [
        " ".join(
            str(part or "")
            for part in (book.title, book.author, book.publisher)
        )
        for book in top_books
    ]
    dedupe_keys = [_dedupe_key(text) for text in top_texts]
    duplicate_free = len(dedupe_keys) == len(set(dedupe_keys))
    expected_term_found = any(
        _contains_any(text, case.expected_terms)
        for text in top_texts
    )
    excluded_term_absent = not any(
        _contains_any(text, case.excluded_terms)
        for text in top_texts
    )
    summary = response.summary or {}
    basis = list(summary.get("recommendationBasis") or [])
    rank_reasons = summary.get("rankReasons") or []
    rank_reasons_present = bool(rank_reasons) and all(
        item.get("rankReason") for item in rank_reasons[: len(top_books)]
    )
    failure_reasons = []
    if response.intent != "BOOK_RECOMMENDATION":
        failure_reasons.append("wrong_intent")
    if response.fallbackUsed:
        failure_reasons.append("fallback_used")
    if not top_books:
        failure_reasons.append("no_results")
    if "semantic" not in basis:
        failure_reasons.append("missing_semantic_basis")
    if not duplicate_free:
        failure_reasons.append("duplicate_top5")
    if not expected_term_found:
        failure_reasons.append("expected_term_not_found")
    if not excluded_term_absent:
        failure_reasons.append("excluded_term_found")
    if not rank_reasons_present:
        failure_reasons.append("missing_rank_reasons")

    return RecommendationCaseResult(
        name=case.name,
        message=case.message,
        passed=not failure_reasons,
        intent=response.intent,
        fallback_used=response.fallbackUsed,
        result_count=response.resultCount,
        recommendation_basis=basis,
        duplicate_free=duplicate_free,
        expected_term_found=expected_term_found,
        excluded_term_absent=excluded_term_absent,
        rank_reasons_present=rank_reasons_present,
        top_titles=[book.title for book in top_books],
        failure_reasons=failure_reasons,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Library Agent book recommendation quality.")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    if not terms:
        return False
    normalized = text.lower()
    return any(term.lower() in normalized for term in terms)


def _dedupe_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


if __name__ == "__main__":
    sys.exit(main())
