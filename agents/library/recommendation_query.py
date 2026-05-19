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
    expected_kdc: tuple[str, ...] = ()


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
    if _looks_like_fiction_reading_request(normalized, subject):
        search_queries = ("장편소설", "소설집", "한국소설", "세계문학", "소설")
        return RecommendationQuery(
            raw_keyword=keyword,
            display_keyword="읽을 만한 소설 작품",
            search_queries=search_queries,
            mode="fiction_reading",
            rewritten=True,
            expected_kdc=("800",),
        )
    if _looks_like_fiction_study_request(normalized, subject):
        search_queries = ("소설 작법", "소설 강의", "문학 비평", "소설 연구", "문학 연구")
        return RecommendationQuery(
            raw_keyword=keyword,
            display_keyword="소설 작법/문학 연구 도서",
            search_queries=search_queries,
            mode="fiction_study",
            rewritten=True,
            expected_kdc=("800",),
        )
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
                expected_kdc=_expected_kdc_for(display_keyword, search_queries, mode),
            )
    cleaned = subject
    return RecommendationQuery(
        raw_keyword=keyword,
        display_keyword=cleaned or keyword,
        search_queries=(cleaned or keyword,),
        mode="topic",
        rewritten=False,
        expected_kdc=_expected_kdc_for(cleaned or keyword, (cleaned or keyword,), "topic"),
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
    for token in ("책", "도서", "추천", "재밌는", "재미있는", "뭐야", "무엇", "알려줘"):
        cleaned = cleaned.replace(token, " ")
    return " ".join(cleaned.split())


def _strip_popularity_keyword(keyword: str) -> str:
    cleaned = keyword.strip()
    for token in (
        "베스트셀러",
        "베스트",
        "인기있는",
        "인기 있는",
        "인기",
        "유명한",
        "유명",
        "가장",
        "제일",
        "많이 읽는",
        "많이 읽힌",
        "뭐야",
        "무엇",
        "알려줘",
    ):
        cleaned = cleaned.replace(token, " ")
    cleaned = " ".join(
        part
        for part in cleaned.split()
        if part not in {"은", "는", "이", "가", "책", "도서"}
    )
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


def _looks_like_fiction_reading_request(normalized: str, subject: str) -> bool:
    if "소설" not in normalized and "문학" not in normalized:
        return False
    if _looks_like_fiction_study_request(normalized, subject):
        return False
    return any(trigger in normalized for trigger in ("추천", "읽을", "읽기", "재밌", "가볍", "볼만"))


def _looks_like_fiction_study_request(normalized: str, subject: str) -> bool:
    if "소설" not in normalized and "문학" not in normalized and "소설" not in subject:
        return False
    return any(trigger in normalized for trigger in ("작법", "쓰기", "쓰는 법", "강의", "연구", "비평", "이론", "장르론"))


def _dedupe_queries(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    queries = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            queries.append(normalized)
    return tuple(queries)


def _expected_kdc_for(display_keyword: str, search_queries: tuple[str, ...], mode: str) -> tuple[str, ...]:
    text = " ".join((display_keyword, *search_queries)).lower()
    if mode == "popular":
        return ()
    rules: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("파이썬", "python", "프로그래밍", "코딩", "인공지능", "ai", "머신러닝", "딥러닝", "데이터", "컴퓨터", "네트워크", "데이터베이스", "database", "sql"), ("000",)),
        (("철학", "심리", "심리학", "상담", "마음"), ("100",)),
        (("종교", "신학", "불교", "기독교"), ("200",)),
        (("경제", "경영", "회계", "재무", "금융", "마케팅", "투자", "통계", "사회", "법", "정치"), ("300",)),
        (("수학", "물리", "화학", "생물", "과학"), ("400",)),
        (("기술", "공학", "건축", "의학", "요리", "패션"), ("500",)),
        (("예술", "디자인", "음악", "미술", "사진", "영화", "만화"), ("600",)),
        (("어학", "영어", "일본어", "중국어", "한국어"), ("700",)),
        (("소설", "문학", "에세이", "시집", "가볍게 읽"), ("800",)),
        (("역사", "여행", "지리"), ("900",)),
    )
    expected: list[str] = []
    for keywords, kdcs in rules:
        if any(keyword in text for keyword in keywords):
            expected.extend(kdcs)
    seen: set[str] = set()
    return tuple(kdc for kdc in expected if not (kdc in seen or seen.add(kdc)))
