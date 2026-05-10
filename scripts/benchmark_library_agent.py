from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Literal, Sequence

from sqlalchemy import text

from agents.library.classifier import (
    IntentClassification,
    RetrievalEvidence,
    classify_intent,
    extract_search_keyword,
    score_intents,
)
from agents.library.db import create_library_engine
from agents.library.generator import build_guide_response
from agents.library.models import BookRecord, GuideChunkRecord, GuideDocRecord
from agents.library.repository import PostgresLibraryRepository
from agents.library.retrieval import (
    GuideContext,
    GuideRetriever,
    GuideSearchResult,
    _dedupe_chunks,
    _dedupe_docs,
    _dedupe_guide_docs,
    _docs_from_chunks,
    _expand_keyword_queries,
    _expand_terms,
    _lexical_boost,
    _rerank_chunks,
    collect_retrieval_evidence,
)
from agents.main_agent.embedding import get_embedding_provider
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.llm.mock import MockLLMClient


StageName = Literal[
    "book_search_sql",
    "guide_chunk_ranking",
    "intent_classification",
    "guide_answer_generation",
]
VariantName = Literal["baseline", "candidate"]


@dataclass(frozen=True)
class BookQuestion:
    question: str
    expected_intent: str
    benchmark_keyword: str
    expected_fields: tuple[str, ...]
    require_location_fields: bool = False


@dataclass(frozen=True)
class GuideQuestion:
    question: str
    expected_intent: str
    expected_title_terms: tuple[str, ...]
    expected_content_terms: tuple[str, ...]


@dataclass(frozen=True)
class IntentQuestion:
    question: str
    expected_intent: str


@dataclass(frozen=True)
class TrialResult:
    stage: StageName
    variant: VariantName
    question: str
    intent: str
    latency_ms: float
    top_result: str | None
    quality_pass: bool
    notes: str
    extra: dict[str, object]


@dataclass(frozen=True)
class AggregatedResult:
    stage: StageName
    variant: VariantName
    question: str
    intent: str
    latency_ms: dict[str, float]
    top_result: str | None
    quality_pass: float
    notes: str


BOOK_QUESTIONS: tuple[BookQuestion, ...] = (
    BookQuestion("파이썬 책 찾아줘", "BOOK_SEARCH", "파이썬", ("title",)),
    BookQuestion("슈카친구들 저자 책 있어?", "BOOK_SEARCH", "슈카친구들", ("author",)),
    BookQuestion("Facet 출판사 책 찾아줘", "BOOK_SEARCH", "Facet", ("publisher",)),
    BookQuestion("청구기호 001 ㄱ785ㄷ 위치 알려줘", "BOOK_LOCATION", "001 ㄱ785ㄷ", ("holding_call_no",), True),
    BookQuestion("인문자연과학자료실 책 찾아줘", "BOOK_SEARCH", "인문자연과학자료실", ("stack_location",), True),
    BookQuestion("1-A-1-d 서가 책 찾아줘", "BOOK_SEARCH", "1-A-1-d", ("stack_shelf",), True),
)

GUIDE_QUESTIONS: tuple[GuideQuestion, ...] = (
    GuideQuestion(
        "도서관 운영 시간 알려줘",
        "LIBRARY_GUIDE",
        ("개관시간", "이용시간", "운영 시간", "휴관일"),
        ("개관시간", "운영시간", "이용시간", "휴관일"),
    ),
    GuideQuestion(
        "오늘 몇 시까지 열어요?",
        "LIBRARY_GUIDE",
        ("개관시간", "이용시간", "운영 시간", "휴관일"),
        ("개관시간", "운영시간", "이용시간", "열람실"),
    ),
    GuideQuestion(
        "책 늦게 반납하면 어떻게 돼?",
        "LIBRARY_GUIDE",
        ("대출", "반납", "연체"),
        ("연체", "반납", "연체료"),
    ),
    GuideQuestion(
        "전자자료 어디서 이용해?",
        "LIBRARY_GUIDE",
        ("전자자료", "전자정보", "RISS", "DB"),
        ("전자자료", "전자정보", "DB", "RISS"),
    ),
    GuideQuestion(
        "좌석 예약 어떻게 해?",
        "LIBRARY_GUIDE",
        ("좌석", "예약", "열람실"),
        ("좌석", "예약", "열람실"),
    ),
)

INTENT_QUESTIONS: tuple[IntentQuestion, ...] = (
    IntentQuestion("도서관에 파이썬 책 있어?", "BOOK_SEARCH"),
    IntentQuestion("도서관 운영 시간", "LIBRARY_GUIDE"),
    IntentQuestion("청구기호 위치", "BOOK_LOCATION"),
    IntentQuestion("궁금한 게 있어", "LIBRARY_GENERAL"),
    IntentQuestion("파이썬 책 찾아줘", "BOOK_SEARCH"),
    IntentQuestion("김코딩 저자 책 있어?", "BOOK_SEARCH"),
    IntentQuestion("Facet 출판사 책 검색", "BOOK_SEARCH"),
    IntentQuestion("도서관에 데이터베이스 관련 책 있어?", "BOOK_SEARCH"),
    IntentQuestion("추천 말고 파이썬 책만 찾아줘", "BOOK_SEARCH"),
    IntentQuestion("청구기호 001 ㄱ785ㄷ 위치 알려줘", "BOOK_LOCATION"),
    IntentQuestion("이 책 어디 있어?", "BOOK_LOCATION"),
    IntentQuestion("제1자료실 책 위치 알려줘", "BOOK_LOCATION"),
    IntentQuestion("1-A-1-d 서가 책 찾아줘", "BOOK_LOCATION"),
    IntentQuestion("도서관 오늘 몇 시까지 열어요?", "LIBRARY_GUIDE"),
    IntentQuestion("책 늦게 반납하면 어떻게 돼?", "LIBRARY_GUIDE"),
    IntentQuestion("전자자료 어디서 이용해?", "LIBRARY_GUIDE"),
    IntentQuestion("좌석 예약 어떻게 해?", "LIBRARY_GUIDE"),
    IntentQuestion("휴관일 언제야?", "LIBRARY_GUIDE"),
    IntentQuestion("이용 문의는 어디에 해?", "LIBRARY_GUIDE"),
    IntentQuestion("안녕하세요", "LIBRARY_GENERAL"),
    IntentQuestion("도와줘", "LIBRARY_GENERAL"),
    IntentQuestion("문의하고 싶은 게 있어", "LIBRARY_GENERAL"),
    IntentQuestion("뭘 물어봐야 할지 모르겠어", "LIBRARY_GENERAL"),
    IntentQuestion("도서관에서 볼만한 파이썬 책 추천해줘", "BOOK_RECOMMENDATION"),
    IntentQuestion("도서관에 파이썬 책 어디 있어?", "BOOK_LOCATION"),
    IntentQuestion("김코딩 책 위치도 알려줘", "BOOK_LOCATION"),
    IntentQuestion("Facet 출판사 책 위치 알려줘", "BOOK_LOCATION"),
    IntentQuestion("추천 말고 위치만 알려줘", "BOOK_LOCATION"),
    IntentQuestion("대출한 책 연장 가능해?", "LIBRARY_GUIDE"),
    IntentQuestion("열람실 자리 예약하고 싶은데", "LIBRARY_GUIDE"),
    IntentQuestion("시험기간에도 오늘 몇 시까지 해?", "LIBRARY_GUIDE"),
    IntentQuestion("전자책 말고 종이책 찾아줘", "BOOK_SEARCH"),
    IntentQuestion("도서관 문의 말고 책 검색하고 싶어", "BOOK_SEARCH"),
    IntentQuestion("뭘 도와줄 수 있어?", "LIBRARY_GENERAL"),
    IntentQuestion("아무거나 좀 물어봐도 돼?", "LIBRARY_GENERAL"),
)

QUALITY_PRIORITY = {
    "book_search_sql": "quality_priority_1",
    "guide_chunk_ranking": "quality_priority_2",
    "intent_classification": "quality_priority_3",
    "guide_answer_generation": "quality_priority_4",
}


class RecordingLLMClient:
    def __init__(self, answer: str = "벤치마크 답변입니다."):
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class CandidateBookSearchRepository:
    def __init__(self, engine):
        self._engine = engine

    def search_books(self, keyword: str, *, location_question: bool, limit: int = 5) -> list[BookRecord]:
        safe_keyword = _escape_like(keyword)
        if not safe_keyword:
            return []

        query = text(
            """
            SELECT
              id,
              bib_no,
              reg_no,
              title,
              author,
              publisher,
              publish_year,
              holding_call_no,
              material_type,
              location_symbol,
              stack_location,
              stack_shelf,
              (
                CASE
                  WHEN lower(holding_call_no) = lower(:keyword) THEN 9.0
                  WHEN holding_call_no ILIKE :prefix ESCAPE '\\' THEN 7.0
                  WHEN holding_call_no ILIKE :pattern ESCAPE '\\' THEN 5.0
                  ELSE 0
                END
                + CASE
                  WHEN lower(stack_shelf) = lower(:keyword) THEN 8.0
                  WHEN stack_shelf ILIKE :prefix ESCAPE '\\' THEN 6.0
                  WHEN stack_shelf ILIKE :pattern ESCAPE '\\' THEN 4.0
                  ELSE 0
                END
                + CASE WHEN title ILIKE :pattern ESCAPE '\\' THEN 5.0 ELSE 0 END
                + CASE WHEN author ILIKE :pattern ESCAPE '\\' THEN 4.0 ELSE 0 END
                + CASE WHEN publisher ILIKE :pattern ESCAPE '\\' THEN 3.0 ELSE 0 END
                + CASE WHEN :location_question AND stack_location ILIKE :pattern ESCAPE '\\' THEN 5.0 ELSE 0 END
                + CASE
                  WHEN :location_question
                   AND holding_call_no IS NOT NULL
                   AND stack_location IS NOT NULL
                   AND stack_shelf IS NOT NULL
                  THEN 1.0
                  ELSE 0
                END
              ) AS relevance_score
            FROM library.books
            WHERE title ILIKE :pattern ESCAPE '\\'
               OR author ILIKE :pattern ESCAPE '\\'
               OR publisher ILIKE :pattern ESCAPE '\\'
               OR holding_call_no ILIKE :pattern ESCAPE '\\'
               OR stack_location ILIKE :pattern ESCAPE '\\'
               OR stack_shelf ILIKE :pattern ESCAPE '\\'
            ORDER BY relevance_score DESC, title ASC, id ASC
            LIMIT :limit
            """
        )
        with self._engine.connect() as connection:
            rows = connection.execute(
                query,
                {
                    "keyword": keyword.strip(),
                    "pattern": f"%{safe_keyword}%",
                    "prefix": f"{safe_keyword}%",
                    "location_question": location_question,
                    "limit": limit,
                },
            ).mappings()
            return [BookRecord(**{key: row[key] for key in BookRecord.__dataclass_fields__}) for row in rows]


class LegacyBookSearchRepository:
    def __init__(self, engine):
        self._engine = engine

    def search_books(self, keyword: str, limit: int = 5) -> list[BookRecord]:
        safe_keyword = _escape_like(keyword)
        if not safe_keyword:
            return []

        query = text(
            """
            SELECT
              id,
              bib_no,
              reg_no,
              title,
              author,
              publisher,
              publish_year,
              holding_call_no,
              material_type,
              location_symbol,
              stack_location,
              stack_shelf
            FROM library.books
            WHERE title ILIKE :pattern ESCAPE '\\'
               OR author ILIKE :pattern ESCAPE '\\'
               OR publisher ILIKE :pattern ESCAPE '\\'
               OR holding_call_no ILIKE :pattern ESCAPE '\\'
               OR stack_location ILIKE :pattern ESCAPE '\\'
               OR stack_shelf ILIKE :pattern ESCAPE '\\'
            ORDER BY title ASC, id ASC
            LIMIT :limit
            """
        )
        with self._engine.connect() as connection:
            rows = connection.execute(
                query,
                {"pattern": f"%{safe_keyword}%", "limit": limit},
            ).mappings()
            return [BookRecord(**row) for row in rows]


def main() -> None:
    args = _parse_args()
    engine = create_library_engine()
    repository = PostgresLibraryRepository(engine)
    legacy_book_repository = LegacyBookSearchRepository(engine)
    embedding_provider = get_embedding_provider()

    stage_results: dict[StageName, list[AggregatedResult]] = {}
    raw_trials: dict[StageName, list[TrialResult]] = {}

    if args.stage in ("all", "book_search_sql"):
        trials = _benchmark_book_search_sql(legacy_book_repository, repository, repeats=args.repeats)
        raw_trials["book_search_sql"] = trials
        stage_results["book_search_sql"] = _aggregate_trials(trials)

    if args.stage in ("all", "guide_chunk_ranking"):
        trials = _benchmark_guide_chunk_ranking(repository, embedding_provider, repeats=args.repeats)
        raw_trials["guide_chunk_ranking"] = trials
        stage_results["guide_chunk_ranking"] = _aggregate_trials(trials)

    if args.stage in ("all", "intent_classification"):
        trials = _benchmark_intent_classification(repository, embedding_provider, repeats=args.repeats)
        raw_trials["intent_classification"] = trials
        stage_results["intent_classification"] = _aggregate_trials(trials)

    if args.stage in ("all", "guide_answer_generation"):
        trials = _benchmark_guide_answer_generation(repository, embedding_provider, repeats=args.repeats)
        raw_trials["guide_answer_generation"] = trials
        stage_results["guide_answer_generation"] = _aggregate_trials(trials)

    pipeline_snapshot = _benchmark_pipeline_snapshot(repository, embedding_provider)
    markdown_summary = _build_markdown_summary(stage_results, pipeline_snapshot)

    payload = {
        "repeats": args.repeats,
        "stage": args.stage,
        "pipeline_snapshot": pipeline_snapshot,
        "aggregated_results": {
            stage: [asdict(result) for result in results]
            for stage, results in stage_results.items()
        },
        "raw_trial_count": {
            stage: len(trials)
            for stage, trials in raw_trials.items()
        },
        "recommendations": {
            "quality_tuning_priority": [
                "book_search_sql",
                "guide_chunk_ranking",
                "intent_classification",
                "guide_answer_generation",
            ],
            "latency_optimization_priority": [
                "evidence_probe_result_reuse",
                "guide_retrieval_duplicate_removal",
                "embedding_query_reuse",
                "sql_index_and_ordering_tuning",
            ],
        },
        "markdown_summary": markdown_summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Library Agent quality bottlenecks.")
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "book_search_sql",
            "guide_chunk_ranking",
            "intent_classification",
            "guide_answer_generation",
        ),
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def _benchmark_book_search_sql(
    repository: PostgresLibraryRepository,
    candidate_repository: CandidateBookSearchRepository,
    *,
    repeats: int,
) -> list[TrialResult]:
    trials: list[TrialResult] = []
    for variant in ("baseline", "candidate"):
        for case in BOOK_QUESTIONS:
            for _ in range(repeats):
                keyword = case.benchmark_keyword
                location_question = case.expected_intent == "BOOK_LOCATION" or _looks_like_location_question(
                    case.question
                )
                started = perf_counter()
                if variant == "baseline":
                    books = repository.search_books(keyword, limit=5)
                else:
                    books = candidate_repository.search_books(
                        keyword,
                        location_question=location_question,
                        limit=5,
                    )
                elapsed = (perf_counter() - started) * 1000
                quality_pass, notes = _evaluate_book_quality(
                    books,
                    keyword=keyword,
                    expected_fields=case.expected_fields,
                    require_location_fields=case.require_location_fields,
                )
                top_result = books[0].title if books else None
                trials.append(
                    TrialResult(
                        stage="book_search_sql",
                        variant=variant,
                        question=case.question,
                        intent=case.expected_intent,
                        latency_ms=elapsed,
                        top_result=top_result,
                        quality_pass=quality_pass,
                        notes=notes,
                        extra={"keyword": keyword},
                    )
                )
    return trials


def _benchmark_guide_chunk_ranking(
    repository: PostgresLibraryRepository,
    embedding_provider,
    *,
    repeats: int,
) -> list[TrialResult]:
    trials: list[TrialResult] = []
    retriever = GuideRetriever(repository, embedding_provider=embedding_provider)
    for variant in ("baseline", "candidate"):
        for case in GUIDE_QUESTIONS:
            for _ in range(repeats):
                keyword = extract_search_keyword(case.question, case.expected_intent)
                started = perf_counter()
                result = (
                    retriever.retrieve(keyword, limit=3)
                    if variant == "baseline"
                    else _retrieve_guide_candidate(keyword, repository, embedding_provider, limit=3)
                )
                elapsed = (perf_counter() - started) * 1000
                top_result = result.chunks[0].title if result.chunks else None
                quality_pass, notes = _evaluate_guide_quality(result, case)
                trials.append(
                    TrialResult(
                        stage="guide_chunk_ranking",
                        variant=variant,
                        question=case.question,
                        intent=case.expected_intent,
                        latency_ms=elapsed,
                        top_result=top_result,
                        quality_pass=quality_pass,
                        notes=notes,
                        extra={"chunk_count": len(result.chunks)},
                    )
                )
    return trials


def _benchmark_intent_classification(
    repository: PostgresLibraryRepository,
    embedding_provider,
    *,
    repeats: int,
) -> list[TrialResult]:
    trials: list[TrialResult] = []
    book_retriever = GuideRetriever(repository, embedding_provider=embedding_provider)
    for variant in ("baseline", "candidate"):
        for case in INTENT_QUESTIONS:
            for _ in range(repeats):
                keyword = extract_search_keyword(case.question)
                started = perf_counter()
                evidence = _collect_evidence(keyword, repository, embedding_provider)
                if variant == "baseline":
                    classification = classify_intent(case.question, evidence=evidence)
                else:
                    classification = _classify_intent_candidate(case.question, evidence)
                elapsed = (perf_counter() - started) * 1000
                quality_pass = classification.intent == case.expected_intent
                notes = f"predicted={classification.intent}, expected={case.expected_intent}"
                trials.append(
                    TrialResult(
                        stage="intent_classification",
                        variant=variant,
                        question=case.question,
                        intent=classification.intent,
                        latency_ms=elapsed,
                        top_result=classification.intent,
                        quality_pass=quality_pass,
                        notes=notes,
                        extra={"confidence": classification.confidence},
                    )
                )
    return trials


def _benchmark_guide_answer_generation(
    repository: PostgresLibraryRepository,
    embedding_provider,
    *,
    repeats: int,
) -> list[TrialResult]:
    trials: list[TrialResult] = []
    guide_retriever = GuideRetriever(repository, embedding_provider=embedding_provider)
    contexts = {
        case.question: guide_retriever.build_context(
            guide_retriever.retrieve(extract_search_keyword(case.question, case.expected_intent), limit=3)
        )
        for case in GUIDE_QUESTIONS
    }
    for variant in ("baseline", "candidate"):
        for case in GUIDE_QUESTIONS:
            for _ in range(repeats):
                context = contexts[case.question]
                llm_client = RecordingLLMClient()
                started = perf_counter()
                if variant == "baseline":
                    response = build_guide_response(
                        case.expected_intent,
                        context,
                        confidence=0.8,
                        llm_client=llm_client,
                    )
                    prompt = llm_client.prompts[0] if llm_client.prompts else ""
                else:
                    response, prompt = _build_candidate_guide_answer_response(
                        case.expected_intent,
                        context,
                        confidence=0.8,
                        llm_client=llm_client,
                    )
                elapsed = (perf_counter() - started) * 1000
                quality_pass = bool(response.answer.strip()) and bool(response.sources)
                notes = "question-specific-guidance=on" if _prompt_has_question_guidance(prompt, case.question) else "question-specific-guidance=off"
                trials.append(
                    TrialResult(
                        stage="guide_answer_generation",
                        variant=variant,
                        question=case.question,
                        intent=case.expected_intent,
                        latency_ms=elapsed,
                        top_result=response.sources[0].title if response.sources else None,
                        quality_pass=quality_pass,
                        notes=notes,
                        extra={"prompt_length": len(prompt)},
                    )
                )
    return trials


def _collect_evidence(keyword: str, repository: PostgresLibraryRepository, embedding_provider) -> RetrievalEvidence:
    guide_retriever = GuideRetriever(repository, embedding_provider=embedding_provider)
    from agents.library.retrieval import BookRetriever

    book_retriever = BookRetriever(repository)
    return collect_retrieval_evidence(
        keyword,
        book_retriever=book_retriever,
        guide_retriever=guide_retriever,
    )


def _classify_intent_candidate(message: str, evidence: RetrievalEvidence) -> IntentClassification:
    scores = score_intents(message, evidence)
    adjusted = {score.intent: score.confidence for score in scores}
    normalized = message.lower()
    location_keywords = ("위치", "어디", "서가", "자료실", "청구기호")
    guide_keywords = ("운영", "휴관", "연체", "예약", "전자자료", "열람실", "도서관")

    if any(keyword in normalized for keyword in location_keywords) and evidence.book_hits:
        adjusted["BOOK_LOCATION"] = adjusted.get("BOOK_LOCATION", 0.0) + 0.12

    if any(keyword in normalized for keyword in guide_keywords) and evidence.guide_hits:
        adjusted["LIBRARY_GUIDE"] = adjusted.get("LIBRARY_GUIDE", 0.0) + 0.10

    if "도서관" in normalized and any(keyword in normalized for keyword in ("운영", "휴관", "연체", "예약")):
        adjusted["BOOK_SEARCH"] = adjusted.get("BOOK_SEARCH", 0.0) - 0.18

    best_intent = max(adjusted, key=adjusted.get)
    confidence = round(max(0.0, min(0.99, adjusted[best_intent])), 3)
    return IntentClassification(
        intent=best_intent,  # type: ignore[arg-type]
        confidence=confidence,
        reason=f"candidate adjusted from evidence for message='{message}'",
        evidence={"bookHits": evidence.book_hits, "guideHits": evidence.guide_hits},
        ambiguous=False,
        used_llm=False,
    )


def _retrieve_guide_candidate(
    keyword: str,
    repository: PostgresLibraryRepository,
    embedding_provider,
    *,
    limit: int,
) -> GuideSearchResult:
    candidates: list[GuideChunkRecord] = []
    for query in _expand_keyword_queries(keyword):
        candidates.extend(repository.search_guide_chunks_by_keyword(query, limit=limit * 4))

    embedding = embedding_provider.embed_query(keyword)
    candidates.extend(
        repository.search_guide_chunks_by_embedding(
            embedding,
            limit=limit * 4,
            min_score=0.35,
        )
    )

    reranked = _rerank_candidate_chunks(keyword, candidates)
    chunks = _dedupe_guide_docs(_dedupe_chunks(reranked))[:limit]
    docs = repository.search_guide_docs(keyword, limit=limit)
    docs = _dedupe_docs([*_docs_from_chunks(chunks), *docs])
    return GuideSearchResult(keyword=keyword, docs=docs, chunks=chunks)


def _rerank_candidate_chunks(keyword: str, chunks: list[GuideChunkRecord]) -> list[GuideChunkRecord]:
    terms = _expand_terms(keyword)
    reranked: list[GuideChunkRecord] = []
    for chunk in chunks:
        score = chunk.score + _lexical_boost(terms, chunk)
        section_line = _first_section_line(chunk.content)
        if section_line and any(term in section_line.lower() for term in terms):
            score += 0.18
        if _looks_table_heavy(chunk.content):
            score -= 0.06
        if _looks_stub_chunk(chunk.content):
            score -= 0.08
        if any(term in {"연체", "반납"} for term in terms) and section_line and any(
            term in section_line.lower() for term in ("연체", "반납")
        ):
            score += 0.12
        if any(term in {"운영", "시간", "개관", "휴관"} for term in terms) and section_line and any(
            term in section_line.lower() for term in ("개관", "시간", "휴관")
        ):
            score += 0.12
        reranked.append(GuideChunkRecord(**{**asdict(chunk), "score": score}))
    return sorted(reranked, key=lambda item: item.score, reverse=True)


def _build_candidate_guide_answer_response(
    intent: str,
    context: GuideContext,
    *,
    confidence: float,
    llm_client: RecordingLLMClient,
):
    prompt = _build_candidate_guide_prompt(context)
    answer = llm_client.generate(prompt)
    response = build_guide_response(
        intent,  # type: ignore[arg-type]
        context,
        confidence,
        llm_client=MockLLMClient(answer),
    )
    return response, prompt


def _build_candidate_guide_prompt(context: GuideContext) -> str:
    guidance = _question_type_guidance(context.keyword)
    blocks = []
    for index, chunk in enumerate(context.chunks[:3], start=1):
        blocks.append(
            f"[{index}] title: {chunk.title}\n"
            f"source_url: {chunk.source_url or '출처 URL 없음'}\n"
            f"score: {chunk.score:.3f}\n"
            f"evidence:\n{_compress_content(chunk.content, max_length=700)}"
        )
    return (
        "당신은 한성대학교 학술정보관 안내를 담당하는 AI입니다.\n"
        "검색 근거에 있는 내용만 사용해 한국어로 답변하세요.\n"
        "모르면 확인할 수 없다고 답하세요.\n"
        f"{guidance}\n\n"
        f"사용자 질문: {context.keyword}\n\n"
        "검색 근거:\n"
        + "\n\n".join(blocks)
    )


def _question_type_guidance(question: str) -> str:
    normalized = question.lower()
    if any(term in normalized for term in ("운영", "시간", "열어요", "휴관")):
        return "답변 형식: 운영시간, 예외 시간, 휴관 유의사항 순서로 짧게 정리하세요."
    if any(term in normalized for term in ("연체", "반납", "늦게")):
        return "답변 형식: 연체 발생 내용, 금액/제재, 예외 또는 확인 필요 사항 순서로 답하세요."
    if any(term in normalized for term in ("전자자료", "db", "전자책", "riss")):
        return "답변 형식: 이용 경로, 로그인 필요 여부, 추가 확인 포인트 순서로 답하세요."
    return "답변 형식: 핵심 답변 먼저, 필요한 보충 설명은 1~2문장으로 제한하세요."


def _aggregate_trials(trials: Sequence[TrialResult]) -> list[AggregatedResult]:
    grouped: dict[tuple[StageName, VariantName, str], list[TrialResult]] = {}
    for trial in trials:
        grouped.setdefault((trial.stage, trial.variant, trial.question), []).append(trial)

    aggregated: list[AggregatedResult] = []
    for (stage, variant, question), items in sorted(grouped.items()):
        latencies = [item.latency_ms for item in items]
        passes = [item.quality_pass for item in items]
        aggregated.append(
            AggregatedResult(
                stage=stage,
                variant=variant,
                question=question,
                intent=items[0].intent,
                latency_ms={
                    "cold": round(latencies[0], 2),
                    "mean": round(statistics.fmean(latencies), 2),
                    "p50": round(_percentile(latencies, 50), 2),
                    "p95": round(_percentile(latencies, 95), 2),
                },
                top_result=items[-1].top_result,
                quality_pass=round(sum(1 for value in passes if value) / len(passes), 3),
                notes=items[-1].notes,
            )
        )
    return aggregated


def _benchmark_pipeline_snapshot(repository: PostgresLibraryRepository, embedding_provider) -> list[dict[str, object]]:
    from agents.library.retrieval import BookRetriever

    book_retriever = BookRetriever(repository)
    guide_retriever = GuideRetriever(repository, embedding_provider=embedding_provider)
    snapshots: list[dict[str, object]] = []
    for question in (
        "파이썬 책 찾아줘",
        "청구기호 001 ㄱ785ㄷ 위치 알려줘",
        "오늘 몇 시까지 열어요?",
        "책 늦게 반납하면 어떻게 돼?",
    ):
        started = perf_counter()
        keyword = extract_search_keyword(question)
        evidence = collect_retrieval_evidence(
            keyword,
            book_retriever=book_retriever,
            guide_retriever=guide_retriever,
        )
        after_evidence = perf_counter()
        classification = classify_intent(question, evidence=evidence)
        keyword_for_intent = extract_search_keyword(question, classification.intent)
        if classification.intent in ("BOOK_SEARCH", "BOOK_LOCATION", "BOOK_RECOMMENDATION"):
            _ = book_retriever.retrieve(keyword_for_intent)
            end = perf_counter()
        else:
            result = guide_retriever.retrieve(keyword_for_intent, limit=3)
            _ = guide_retriever.build_context(result)
            end = perf_counter()
        snapshots.append(
            {
                "question": question,
                "intent": classification.intent,
                "evidence_ms": round((after_evidence - started) * 1000, 2),
                "post_evidence_ms": round((end - after_evidence) * 1000, 2),
                "total_ms": round((end - started) * 1000, 2),
            }
        )
    return snapshots


def _build_markdown_summary(
    stage_results: dict[StageName, list[AggregatedResult]],
    pipeline_snapshot: list[dict[str, object]],
) -> str:
    lines = [
        "# Library Agent Benchmark Summary",
        "",
        "## Stage Comparison",
        "",
        "| stage | baseline_p50_ms | candidate_p50_ms | delta_ms | quality_before | quality_after | recommended_priority |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for stage, results in stage_results.items():
        baseline = [item for item in results if item.variant == "baseline"]
        candidate = [item for item in results if item.variant == "candidate"]
        baseline_p50 = _mean_metric(baseline, "p50")
        candidate_p50 = _mean_metric(candidate, "p50")
        baseline_quality = round(statistics.fmean(item.quality_pass for item in baseline), 3) if baseline else 0.0
        candidate_quality = round(statistics.fmean(item.quality_pass for item in candidate), 3) if candidate else 0.0
        lines.append(
            f"| {stage} | {baseline_p50:.2f} | {candidate_p50:.2f} | {candidate_p50 - baseline_p50:+.2f} | {baseline_quality:.3f} | {candidate_quality:.3f} | {QUALITY_PRIORITY[stage]} |"
        )

    lines.extend(
        [
            "",
            "## Pipeline Snapshot",
            "",
            "| question | intent | evidence_ms | post_evidence_ms | total_ms |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in pipeline_snapshot:
        lines.append(
            f"| {row['question']} | {row['intent']} | {row['evidence_ms']} | {row['post_evidence_ms']} | {row['total_ms']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 품질 개선 1순위는 `book_search_sql`로 유지한다.",
            "- 속도 개선 1순위는 `collect_retrieval_evidence()`와 branch retrieval의 중복 호출 제거다.",
            "- `intent_classification` 함수 자체는 빠르므로, latency보다는 downstream 오탐 감소 효과를 본다.",
        ]
    )
    return "\n".join(lines)


def _mean_metric(results: Sequence[AggregatedResult], key: str) -> float:
    if not results:
        return 0.0
    return statistics.fmean(item.latency_ms[key] for item in results)


def _evaluate_book_quality(
    books: list[BookRecord],
    *,
    keyword: str,
    expected_fields: tuple[str, ...],
    require_location_fields: bool,
) -> tuple[bool, str]:
    if not books:
        return False, "no_results"

    top = books[0]
    normalized_keyword = _normalize(keyword)
    matched = any(normalized_keyword in _normalize(getattr(top, field_name)) for field_name in expected_fields)
    if not matched:
        return False, "top1_field_mismatch"

    if require_location_fields and not (top.holding_call_no and top.stack_location and top.stack_shelf):
        return False, "missing_location_fields"
    return True, "top1_match"


def _evaluate_guide_quality(result: GuideSearchResult, case: GuideQuestion) -> tuple[bool, str]:
    if not result.chunks:
        return False, "no_chunks"
    top = result.chunks[0]
    title = top.title.lower()
    content = top.content.lower()
    if any(term.lower() in title for term in case.expected_title_terms):
        return True, "title_term_match"
    if any(term.lower() in content for term in case.expected_content_terms):
        return True, "content_term_match"
    return False, "top1_mismatch"


def _prompt_has_question_guidance(prompt: str, question: str) -> bool:
    normalized_question = question.lower()
    if any(term in normalized_question for term in ("운영", "시간", "열어요", "휴관")):
        return "답변 형식: 운영시간, 예외 시간, 휴관 유의사항 순서로 짧게 정리하세요." in prompt
    if any(term in normalized_question for term in ("연체", "반납", "늦게")):
        return "답변 형식: 연체 발생 내용, 금액/제재, 예외 또는 확인 필요 사항 순서로 답하세요." in prompt
    if any(term in normalized_question for term in ("전자자료", "db", "전자책", "riss")):
        return "답변 형식: 이용 경로, 로그인 필요 여부, 추가 확인 포인트 순서로 답하세요." in prompt
    return "답변 형식: 핵심 답변 먼저, 필요한 보충 설명은 1~2문장으로 제한하세요." in prompt


def _first_section_line(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("섹션: "):
            return stripped
    return None


def _looks_table_heavy(content: str) -> bool:
    pipe_count = content.count("|")
    return pipe_count >= 12 and pipe_count > len(content.splitlines()) * 1.5


def _looks_stub_chunk(content: str) -> bool:
    compact = content.strip()
    return len(compact) < 80 or compact.endswith(": 바로가기")


def _looks_like_location_question(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in ("위치", "어디", "서가", "자료실", "청구기호"))


def _compress_content(content: str, *, max_length: int) -> str:
    compact = " ".join(content.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def _escape_like(value: str) -> str:
    return value.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _normalize(value: object) -> str:
    return str(value or "").lower().replace(" ", "")


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    weight = position - lower
    return lower_value + ((upper_value - lower_value) * weight)


if __name__ == "__main__":
    main()
