from __future__ import annotations

from dataclasses import dataclass

from agents.library.classifier import RetrievalEvidence
from agents.library.models import BookRecord, GuideDocRecord
from agents.library.repository import LibraryRepository


@dataclass(frozen=True)
class BookSearchResult:
    keyword: str
    books: list[BookRecord]


@dataclass(frozen=True)
class GuideSearchResult:
    keyword: str
    docs: list[GuideDocRecord]


@dataclass(frozen=True)
class GuideContext:
    keyword: str
    docs: list[GuideDocRecord]
    context_text: str


class BookRetriever:
    def __init__(self, repository: LibraryRepository):
        self._repository = repository

    def retrieve(self, keyword: str, limit: int = 5) -> BookSearchResult:
        books = self._repository.search_books(keyword, limit=limit)
        return BookSearchResult(keyword=keyword, books=books)


class GuideRetriever:
    def __init__(self, repository: LibraryRepository):
        self._repository = repository

    def retrieve(self, keyword: str, limit: int = 3) -> GuideSearchResult:
        docs = self._repository.search_guide_docs(keyword, limit=limit)
        return GuideSearchResult(keyword=keyword, docs=docs)

    def build_context(self, result: GuideSearchResult) -> GuideContext:
        chunks = []
        for index, doc in enumerate(result.docs, start=1):
            chunks.append(f"[{index}] {doc.title}\n{doc.content}")
        return GuideContext(
            keyword=result.keyword,
            docs=result.docs,
            context_text="\n\n".join(chunks),
        )


def collect_retrieval_evidence(
    keyword: str,
    book_retriever: BookRetriever,
    guide_retriever: GuideRetriever,
) -> RetrievalEvidence:
    book_probe = book_retriever.retrieve(keyword, limit=1)
    guide_probe = guide_retriever.retrieve(keyword, limit=1)
    return RetrievalEvidence(
        book_hits=len(book_probe.books),
        guide_hits=len(guide_probe.docs),
        book_keyword=keyword,
        guide_keyword=keyword,
    )

