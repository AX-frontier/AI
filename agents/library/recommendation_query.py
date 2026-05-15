from __future__ import annotations

from dataclasses import dataclass

from agents.main_agent.llm.base import LLMClient


@dataclass(frozen=True)
class RecommendationQuery:
    raw_keyword: str
    display_keyword: str
    search_queries: tuple[str, ...]
    mode: str
    rewritten: bool = False


def interpret_recommendation_query(
    message: str,
    keyword: str,
    *,
    llm_client: LLMClient | None = None,
) -> RecommendationQuery:
    """추천 요청의 자연어 표현을 검색 가능한 주제어로 재작성한다.

    LLM이 없어도 안정적으로 동작해야 하므로 룰 기반 해석을 기본으로 둔다.
    """
    rule_result = _interpret_by_rules(message, keyword)
    if rule_result.rewritten:
        return rule_result
    llm_result = _interpret_by_llm(message, keyword, llm_client)
    return llm_result or rule_result


def _interpret_by_rules(message: str, keyword: str) -> RecommendationQuery:
    normalized = f"{message} {keyword}".lower()
    subject = _strip_generic_keyword(keyword)
    mappings = [
        (
            ("재밌", "재미", "흥미", "가볍", "쉽게 읽", "읽기 쉬", "부담 없"),
            "가볍게 읽을 만한 교양/소설/에세이",
            ("교양", "소설", "에세이", "상식", "여행", "역사"),
            "light_reading",
        ),
        (
            ("쉬운", "쉽", "입문", "처음", "초보", "기초"),
            f"{subject or '입문'} 입문서",
            _introductory_queries(subject),
            "introductory",
        ),
        (
            ("인기", "베스트", "많이 읽", "유명"),
            f"{_strip_popularity_keyword(subject)} 인기 도서"
            if _strip_popularity_keyword(subject)
            else "알라딘 베스트셀러 소장 도서",
            _popular_queries(_strip_popularity_keyword(subject)),
            "popular",
        ),
        (
            ("전공", "교재", "수업", "학과"),
            "전공/수업 참고 도서",
            (f"{subject} 전공", f"{subject} 교재", subject, "전공", "교재") if subject else ("전공", "교재", "개론"),
            "academic",
        ),
    ]
    for triggers, display_keyword, search_queries, mode in mappings:
        if any(trigger in normalized for trigger in triggers):
            return RecommendationQuery(
                raw_keyword=keyword,
                display_keyword=display_keyword,
                search_queries=_dedupe_queries(search_queries),
                mode=mode,
                rewritten=True,
            )
    cleaned = subject
    return RecommendationQuery(
        raw_keyword=keyword,
        display_keyword=cleaned or keyword,
        search_queries=(cleaned or keyword,),
        mode="topic",
        rewritten=False,
    )


def _interpret_by_llm(
    message: str,
    keyword: str,
    llm_client: LLMClient | None,
) -> RecommendationQuery | None:
    # Hook point for a future strict JSON LLM rewriter. Candidate generation must remain DB-only.
    return None


def _strip_generic_keyword(keyword: str) -> str:
    cleaned = keyword.strip()
    for token in ("책", "도서", "추천", "재밌는", "재미있는"):
        cleaned = cleaned.replace(token, " ")
    return " ".join(cleaned.split())


def _strip_popularity_keyword(keyword: str) -> str:
    cleaned = keyword.strip()
    for token in ("베스트셀러", "베스트", "인기", "유명한", "유명"):
        cleaned = cleaned.replace(token, " ")
    return " ".join(cleaned.split())


def _popular_queries(subject: str) -> tuple[str, ...]:
    if subject:
        return (subject, f"{subject} 인기", f"{subject} 베스트셀러")
    return ()


def _introductory_queries(subject: str) -> tuple[str, ...]:
    if subject:
        return (
            f"{subject} 입문",
            f"{subject} 기초",
            f"{subject} 초보",
            subject,
            "입문",
        )
    return ("입문", "기초", "초보")


def _dedupe_queries(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    queries = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            queries.append(normalized)
    return tuple(queries)
