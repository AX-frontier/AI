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

    def __init__(self, repository: LibraryRepository):
        self._repository = repository

    def retrieve(self, keyword: str, limit: int = 5) -> BookSearchResult:
        """도서명, 저자, 출판사, 청구기호, 위치 기준으로 library.books를 검색한다."""
        books = self._repository.search_books(keyword, limit=limit)
        return BookSearchResult(keyword=keyword, books=books)


class GuideRetriever:
    """안내 QA 검색을 담당하는 retriever."""

    def __init__(
        self,
        repository: LibraryRepository,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self._repository = repository
        self._embedding_provider = embedding_provider

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
            embedding = self._get_embedding_provider().embed_query(keyword)
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


def _dedupe_chunks(chunks: list[GuideChunkRecord]) -> list[GuideChunkRecord]:
    by_id: dict[int, GuideChunkRecord] = {}
    for chunk in chunks:
        existing = by_id.get(chunk.id)
        if existing is None or chunk.score > existing.score:
            by_id[chunk.id] = chunk
    return sorted(by_id.values(), key=lambda chunk: chunk.score, reverse=True)


def _rerank_chunks(keyword: str, chunks: list[GuideChunkRecord]) -> list[GuideChunkRecord]:
    terms = _expand_terms(keyword)
    reranked = [
        replace(chunk, score=chunk.score + _lexical_boost(terms, chunk))
        for chunk in chunks
    ]
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
) -> RetrievalEvidence:
    """분류기가 실제 검색 evidence를 쓸 수 있도록 각 저장소를 1건만 조회한다."""
    book_probe = book_retriever.retrieve(keyword, limit=1)
    guide_probe = guide_retriever.retrieve(keyword, limit=1)
    return RetrievalEvidence(
        book_hits=len(book_probe.books),
        guide_hits=len(guide_probe.docs),
        book_keyword=keyword,
        guide_keyword=keyword,
    )
