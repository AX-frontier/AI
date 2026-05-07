from __future__ import annotations

import re
from dataclasses import dataclass, replace

from agents.main_agent.embedding import EmbeddingProvider
from agents.main_agent.models import MainChunkRecord
from agents.main_agent.repository import MainChunkRepository


@dataclass(frozen=True)
class MainSearchResult:
    keyword: str
    chunks: list[MainChunkRecord]


class MainRetriever:
    """학교 공지/학사/일반 안내 chunk 검색을 담당한다."""

    def __init__(self, repository: MainChunkRepository, embedding_provider: EmbeddingProvider):
        self._repository = repository
        self._embedding_provider = embedding_provider

    def retrieve(self, message: str, limit: int = 5) -> MainSearchResult:
        keyword = extract_main_search_keyword(message)
        embedding = self._embedding_provider.embed_query(keyword)
        candidates = self._repository.search_similar_chunks(embedding, limit=max(limit * 6, 20))
        keyword_search = getattr(self._repository, "search_keyword_chunks", None)
        if keyword_search:
            candidates.extend(keyword_search(_expand_terms(keyword), limit=limit * 4))
        candidates = _dedupe_chunks(candidates)
        chunks = _dedupe_documents(rerank_chunks(keyword, candidates))[:limit]
        return MainSearchResult(keyword=keyword, chunks=chunks)


def extract_main_search_keyword(message: str) -> str:
    """질문에서 검색에 불필요한 명령형 표현을 줄인다."""
    cleaned = message.strip()
    cleaned = re.sub(r"[?？!！.。]+$", "", cleaned).strip()
    cleanup_phrases = (
        "알려줘",
        "궁금해",
        "뭐야",
        "언제야",
        "어떻게",
        "확인해줘",
        "찾아줘",
        "검색해줘",
    )
    for phrase in cleanup_phrases:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:;")
    return cleaned or message.strip()


def rerank_chunks(keyword: str, chunks: list[MainChunkRecord]) -> list[MainChunkRecord]:
    """Vector score 후보를 제목/본문 키워드 근거로 보정한다."""
    terms = _expand_terms(keyword)
    reranked = [
        replace(chunk, score=min(0.99, chunk.score + _safe_lexical_boost(keyword, terms, chunk)))
        for chunk in chunks
    ]
    return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)


def _expand_terms(keyword: str) -> list[str]:
    raw_terms = re.findall(r"[0-9A-Za-z가-힣]+", keyword.lower())
    terms: list[str] = []
    for term in raw_terms:
        if len(term) >= 2:
            terms.append(term)
        if "전공" in term:
            terms.append("전공")
        if "복수" in term:
            terms.append("복수")
        if "도서관" in term:
            terms.extend(["도서관", "학술정보관"])
        if "운영" in term or "시간" in term:
            terms.extend(["운영", "시간", "개관"])
    seen: set[str] = set()
    return [term for term in terms if not (term in seen or seen.add(term))]


def _lexical_boost(keyword: str, terms: list[str], chunk: MainChunkRecord) -> float:
    title = chunk.title.lower()
    text = chunk.text.lower()
    category = (chunk.category or "").lower()
    doc_type = str(chunk.metadata.get("doc_type") or "").lower()
    source = str(chunk.metadata.get("source") or "").lower()
    boost = 0.0

    normalized_keyword = re.sub(r"\s+", "", keyword.lower())
    normalized_title = re.sub(r"\s+", "", title)
    normalized_text = re.sub(r"\s+", "", text)
    if normalized_keyword and normalized_keyword in normalized_title:
        boost += 0.18
    elif normalized_keyword and normalized_keyword in normalized_text:
        boost += 0.06

    for term in terms:
        if term in title:
            boost += 0.07
        if term in text:
            boost += 0.018

    if any(term in terms for term in ("전공", "복수", "학사", "수강")):
        if "학사" in category:
            boost += 0.05
        if "전공" in title:
            boost += 0.08

    library_terms = {"도서관", "학술정보관", "운영", "시간", "개관", "휴관"}
    if library_terms.intersection(terms):
        if source == "hsel_library" and doc_type == "pages":
            boost += 0.08
        if any(term in title for term in ("개관", "시간", "휴관")):
            boost += 0.12

    return boost


def _safe_lexical_boost(keyword: str, terms: list[str], chunk: MainChunkRecord) -> float:
    if chunk.score < 0.35:
        return 0.0
    return _lexical_boost(keyword, terms, chunk)


def _dedupe_chunks(chunks: list[MainChunkRecord]) -> list[MainChunkRecord]:
    by_id: dict[str, MainChunkRecord] = {}
    for chunk in chunks:
        existing = by_id.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            by_id[chunk.chunk_id] = chunk
    return list(by_id.values())


def _dedupe_documents(chunks: list[MainChunkRecord]) -> list[MainChunkRecord]:
    by_document_id: dict[str, MainChunkRecord] = {}
    for chunk in chunks:
        existing = by_document_id.get(chunk.document_id)
        if existing is None or _is_better_document_chunk(chunk, existing):
            by_document_id[chunk.document_id] = chunk
    return sorted(by_document_id.values(), key=lambda chunk: chunk.score, reverse=True)


def _is_better_document_chunk(candidate: MainChunkRecord, existing: MainChunkRecord) -> bool:
    if candidate.score > existing.score + 0.02:
        return True
    if existing.score > candidate.score + 0.02:
        return False
    return _chunk_index(candidate) < _chunk_index(existing)


def _chunk_index(chunk: MainChunkRecord) -> int:
    value = chunk.metadata.get("chunk_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 9999
