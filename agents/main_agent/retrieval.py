from __future__ import annotations

import re
from dataclasses import dataclass

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
        chunks = self._repository.search_similar_chunks(embedding, limit=limit)
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
