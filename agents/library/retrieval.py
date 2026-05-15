from __future__ import annotations

import re
from dataclasses import dataclass, replace

from agents.library.classifier import RetrievalEvidence
from agents.library.models import BookRecord, GuideChunkRecord, GuideDocRecord
from agents.library.repository import LibraryRepository
from agents.main_agent.embedding import EmbeddingProvider, get_embedding_provider


@dataclass(frozen=True)
class BookSearchResult:
    """정리된 사용자 검색어로 찾은 도서 목록."""

    keyword: str
    books: list[BookRecord]


@dataclass(frozen=True)
class GuideSearchResult:
    """정리된 사용자 검색어로 찾은 안내 문서 목록."""

    keyword: str
    docs: list[GuideDocRecord]
    chunks: list[GuideChunkRecord]


@dataclass(frozen=True)
class GuideContext:
    """나중에 LLM에 전달할 수 있는 RAG용 안내 문서 컨텍스트."""

    keyword: str
    docs: list[GuideDocRecord]
    chunks: list[GuideChunkRecord]
    context_text: str


class BookRetriever:
    """정형 도서 검색을 담당하는 repository 기반 retriever."""

    def __init__(
        self,
        repository: LibraryRepository,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._query_embedding_cache: dict[str, list[float]] = {}

    def retrieve(
        self,
        keyword: str,
        limit: int = 5,
        *,
        location_question: bool = False,
        include_semantic: bool = False,
    ) -> BookSearchResult:
        """도서명, 저자, 출판사, 청구기호, 위치 기준으로 library.books를 검색한다."""
        books = self._repository.search_books(
            keyword,
            limit=limit,
            location_question=location_question,
        )
        if include_semantic and not location_question:
            books = _dedupe_books(
                [
                    *books,
                    *self._retrieve_semantic_books(keyword, limit=limit),
                ]
            )[:limit]
        return BookSearchResult(keyword=keyword, books=books)

    def _retrieve_semantic_books(self, keyword: str, *, limit: int) -> list[BookRecord]:
        semantic_search = getattr(self._repository, "search_books_by_embedding", None)
        if not semantic_search:
            return []
        try:
            embedding = self._embed_query(keyword)
            return semantic_search(embedding, limit=limit * 2, min_score=0.35)
        except Exception:
            return []

    def _get_embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    def _embed_query(self, keyword: str) -> list[float]:
        normalized = keyword.strip()
        cached = self._query_embedding_cache.get(normalized)
        if cached is not None:
            return cached

        embedding = self._get_embedding_provider().embed_query(normalized)
        self._query_embedding_cache[normalized] = embedding
        return embedding


class GuideRetriever:
    """안내 QA 검색을 담당하는 retriever."""

    def __init__(
        self,
        repository: LibraryRepository,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._query_embedding_cache: dict[str, list[float]] = {}

    def retrieve(self, keyword: str, limit: int = 3) -> GuideSearchResult:
        """키워드 검색과 semantic 검색을 합쳐 관련 안내 chunk를 찾는다."""
        chunk_candidates = self._retrieve_chunks(keyword, limit=limit)
        chunks = _dedupe_guide_docs(_dedupe_chunks(_rerank_chunks(keyword, chunk_candidates)))[:limit]
        docs = self._repository.search_guide_docs(keyword, limit=limit)
        docs = _dedupe_docs([*_docs_from_chunks(chunks), *docs])
        return GuideSearchResult(keyword=keyword, docs=docs, chunks=chunks)

    def build_context(self, result: GuideSearchResult) -> GuideContext:
        """검색된 안내 문서를 RAG 답변 생성용 짧은 컨텍스트로 정리한다."""
        context_parts = []
        for index, chunk in enumerate(result.chunks, start=1):
            context_parts.append(f"[{index}] {chunk.title}\n{chunk.content}")
        if not context_parts:
            for index, doc in enumerate(result.docs, start=1):
                context_parts.append(f"[{index}] {doc.title}\n{doc.content}")
        return GuideContext(
            keyword=result.keyword,
            docs=result.docs,
            chunks=result.chunks,
            context_text="\n\n".join(context_parts),
        )

    def _retrieve_chunks(self, keyword: str, *, limit: int) -> list[GuideChunkRecord]:
        candidates: list[GuideChunkRecord] = []
        keyword_search = getattr(self._repository, "search_guide_chunks_by_keyword", None)
        if keyword_search:
            for query in _expand_keyword_queries(keyword):
                candidates.extend(keyword_search(query, limit=limit * 4))

        semantic_search = getattr(self._repository, "search_guide_chunks_by_embedding", None)
        if semantic_search:
            embedding = self._embed_query(keyword)
            candidates.extend(
                semantic_search(
                    embedding,
                    limit=limit * 4,
                    min_score=0.35,
                )
            )
        return candidates

    def _get_embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    def _embed_query(self, keyword: str) -> list[float]:
        normalized = keyword.strip()
        cached = self._query_embedding_cache.get(normalized)
        if cached is not None:
            return cached

        embedding = self._get_embedding_provider().embed_query(normalized)
        self._query_embedding_cache[normalized] = embedding
        return embedding


def _dedupe_chunks(chunks: list[GuideChunkRecord]) -> list[GuideChunkRecord]:
    by_id: dict[int, GuideChunkRecord] = {}
    for chunk in chunks:
        existing = by_id.get(chunk.id)
        if existing is None or chunk.score > existing.score:
            by_id[chunk.id] = chunk
    return sorted(by_id.values(), key=lambda chunk: chunk.score, reverse=True)


def _dedupe_books(books: list[BookRecord]) -> list[BookRecord]:
    by_id: dict[int, BookRecord] = {}
    for book in books:
        by_id.setdefault(book.id, book)
    return list(by_id.values())


def _rerank_chunks(keyword: str, chunks: list[GuideChunkRecord]) -> list[GuideChunkRecord]:
    terms = _expand_terms(keyword)
    contact_query = _is_staff_contact_query(terms)
    service_contact_query = _is_service_contact_query(terms)
    reranked = []
    for chunk in chunks:
        score = chunk.score + _lexical_boost(terms, chunk)
        section_line = _first_section_line(chunk.content)
        if section_line and any(term in section_line.lower() for term in terms):
            score += 0.18
        if _looks_table_heavy(chunk.content) and not contact_query:
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
        if contact_query:
            score += _staff_contact_boost(terms, chunk)
        if service_contact_query:
            score += _service_contact_boost(terms, chunk)
        reranked.append(replace(chunk, score=score))
    return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)


def _expand_keyword_queries(keyword: str) -> list[str]:
    queries = [keyword.strip()]
    queries.extend(_expand_terms(keyword))
    seen: set[str] = set()
    return [query for query in queries if query and not (query in seen or seen.add(query))]


def _expand_terms(keyword: str) -> list[str]:
    raw_terms = re.findall(r"[0-9A-Za-z가-힣]+", keyword.lower())
    terms: list[str] = []
    for term in raw_terms:
        if len(term) >= 2:
            terms.append(term)
        if "관장" in term:
            terms.extend(["관장", "조직안내", "조직", "직원", "연락처", "이메일", "전화번호"])
        if (
            "직원" in term
            or "조직" in term
            or "부서" in term
            or "담당" in term
            or "팀장" in term
            or "연락" in term
            or "문의" in term
        ):
            terms.extend(["조직안내", "조직", "직원", "담당", "연락처", "이메일", "전화번호"])
        if "메일" in term or "이메일" in term:
            terms.extend(["이메일", "메일", "연락처", "조직안내"])
        if "번호" in term or "전화" in term:
            terms.extend(["전화번호", "전화", "연락처", "조직안내"])
        if term in {"오늘", "몇", "시", "까지"} or "열어" in term or "닫" in term:
            terms.extend(["운영", "시간", "개관", "휴관", "열람실"])
        if "운영" in term or "시간" in term or "개관" in term:
            terms.extend(["운영", "시간", "개관", "휴관"])
        if "반납" in term or "늦" in term or "연체" in term:
            terms.extend(["대출", "반납", "연체"])
        if "예약" in term:
            terms.extend(["예약", "좌석", "시설"])
        if "전자" in term or "db" in term:
            terms.extend(["전자자료", "전자책", "db"])
    seen: set[str] = set()
    return [term for term in terms if term and not (term in seen or seen.add(term))]


def _lexical_boost(terms: list[str], chunk: GuideChunkRecord) -> float:
    title = chunk.title.lower()
    content = chunk.content.lower()
    boost = 0.0
    for term in terms:
        if term in title:
            boost += 0.09
        if term in content:
            boost += 0.018
            boost += min(content.count(term) * 0.012, 0.12)
        if f"섹션: {term}" in content:
            boost += 0.14

    time_terms = {"운영", "시간", "개관", "휴관", "열람실"}
    if time_terms.intersection(terms):
        if "개관" in title or "시간" in title or "휴관" in title:
            boost += 0.18

    lending_terms = {"대출", "반납", "연장", "예약", "연체"}
    if lending_terms.intersection(terms):
        if any(term in title for term in lending_terms):
            boost += 0.18

    return boost


def _is_staff_contact_query(terms: list[str]) -> bool:
    contact_terms = {
        "관장",
        "직원",
        "조직",
        "조직안내",
        "부서",
        "담당",
        "팀장",
        "연락",
        "연락처",
        "문의",
        "이메일",
        "메일",
        "전화",
        "전화번호",
        "번호",
    }
    return bool(contact_terms.intersection(terms))


def _is_service_contact_query(terms: list[str]) -> bool:
    service_terms = {
        "문의",
        "전화",
        "전화번호",
        "번호",
        "연락",
        "연락처",
        "대출",
        "반납",
        "연장",
        "분실",
        "구입",
        "구독",
        "시설",
        "기기",
        "열람실",
        "원문복사",
        "상호대차",
    }
    return bool(service_terms.intersection(terms))


def _staff_contact_boost(terms: list[str], chunk: GuideChunkRecord) -> float:
    title = chunk.title.lower()
    content = chunk.content.lower()
    boost = 0.0

    if "조직안내" in title or "조직 안내" in content or "문서 제목: 조직안내" in content:
        boost += 1.2
    if "직위" in content and "성명" in content and "업무" in content:
        boost += 0.45
    if ("이메일" in content or "메일" in content) and {"이메일", "메일", "연락처"}.intersection(terms):
        boost += 0.28
    if ("전화번호" in content or "전화" in content) and {"전화번호", "전화", "번호", "연락처"}.intersection(terms):
        boost += 0.28
    if "관장" in terms and "관장" in content:
        boost += 0.45

    generic_mail_only = {"이메일", "메일", "전화", "전화번호", "번호"}.intersection(terms)
    if generic_mail_only and "조직안내" not in title and "조직 안내" not in content:
        boost -= 0.18
    explicit_person_terms = {"관장", "담당자", "누구"}
    if "업무별 안내" in content and not explicit_person_terms.intersection(terms):
        boost += 0.35
    if "직위" in content and "성명" in content and not explicit_person_terms.intersection(terms):
        if {"문의", "전화", "전화번호", "번호", "연락처"}.intersection(terms):
            boost -= 0.8

    return boost


def _service_contact_boost(terms: list[str], chunk: GuideChunkRecord) -> float:
    content = chunk.content.lower()
    boost = 0.0
    if "섹션: 업무별 안내" in content:
        boost += 1.1
    if "전화번호" in content and {"전화", "전화번호", "번호", "문의", "연락처"}.intersection(terms):
        boost += 0.35
    if {"대출", "반납", "연장", "분실"}.intersection(terms) and any(
        term in content for term in ("대출", "반납", "연장", "분실")
    ):
        boost += 0.45
    return boost


def _first_section_line(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("섹션:"):
            return stripped
    return None


def _looks_table_heavy(content: str) -> bool:
    lines = [line for line in content.splitlines() if line.strip()]
    table_lines = [line for line in lines if "|" in line]
    return len(table_lines) >= 4 and len(table_lines) >= max(3, len(lines) // 2)


def _looks_stub_chunk(content: str) -> bool:
    normalized = " ".join(line.strip() for line in content.splitlines() if line.strip())
    if len(normalized) < 80:
        return True
    return bool(re.fullmatch(r"[-|:0-9A-Za-z가-힣 ./()]+", normalized)) and normalized.count(".") == 0


def _dedupe_guide_docs(chunks: list[GuideChunkRecord]) -> list[GuideChunkRecord]:
    by_doc_id: dict[int, GuideChunkRecord] = {}
    for chunk in chunks:
        existing = by_doc_id.get(chunk.guide_doc_id)
        if existing is None or _is_better_chunk(chunk, existing):
            by_doc_id[chunk.guide_doc_id] = chunk
    return sorted(by_doc_id.values(), key=lambda chunk: chunk.score, reverse=True)


def _is_better_chunk(candidate: GuideChunkRecord, existing: GuideChunkRecord) -> bool:
    if candidate.score > existing.score + 0.02:
        return True
    if existing.score > candidate.score + 0.02:
        return False
    return candidate.chunk_index < existing.chunk_index


def _docs_from_chunks(chunks: list[GuideChunkRecord]) -> list[GuideDocRecord]:
    return [
        GuideDocRecord(
            id=chunk.guide_doc_id,
            source_url=chunk.source_url,
            title=chunk.title,
            content=chunk.content,
            updated_at=chunk.updated_at,
        )
        for chunk in chunks
    ]


def _dedupe_docs(docs: list[GuideDocRecord]) -> list[GuideDocRecord]:
    by_id: dict[int, GuideDocRecord] = {}
    for doc in docs:
        if doc.id not in by_id:
            by_id[doc.id] = doc
    return list(by_id.values())


def collect_retrieval_evidence(
    keyword: str,
    book_retriever: BookRetriever,
    guide_retriever: GuideRetriever,
    *,
    book_probe_limit: int = 1,
    guide_probe_limit: int = 1,
    book_location_question: bool = False,
    include_probe_results: bool = False,
) -> RetrievalEvidence:
    """분류기가 실제 검색 evidence를 쓸 수 있도록 각 저장소를 1건만 조회한다."""
    book_probe = book_retriever.retrieve(
        keyword,
        limit=book_probe_limit,
        location_question=book_location_question,
    )
    guide_probe = guide_retriever.retrieve(keyword, limit=guide_probe_limit)
    return RetrievalEvidence(
        book_hits=len(book_probe.books),
        guide_hits=len(guide_probe.docs),
        book_keyword=keyword,
        guide_keyword=keyword,
        book_probe_limit=book_probe_limit,
        guide_probe_limit=guide_probe_limit,
        book_probe_location_question=book_location_question,
        book_probe=book_probe if include_probe_results else None,
        guide_probe=guide_probe if include_probe_results else None,
    )
