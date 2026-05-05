from __future__ import annotations

from dataclasses import dataclass

from agents.library.classifier import RetrievalEvidence
from agents.library.models import BookRecord, GuideDocRecord
from agents.library.repository import LibraryRepository


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


@dataclass(frozen=True)
class GuideContext:
    """나중에 LLM에 전달할 수 있는 RAG용 안내 문서 컨텍스트."""

    keyword: str
    docs: list[GuideDocRecord]
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
    """안내 QA 검색을 담당하는 retriever. 이후 pgvector 검색을 이곳에 붙인다."""

    def __init__(self, repository: LibraryRepository):
        self._repository = repository

    def retrieve(self, keyword: str, limit: int = 3) -> GuideSearchResult:
        """검색어와 관련된 안내 문서를 library.guide_docs에서 찾는다."""
        docs = self._repository.search_guide_docs(keyword, limit=limit)
        return GuideSearchResult(keyword=keyword, docs=docs)

    def build_context(self, result: GuideSearchResult) -> GuideContext:
        """검색된 안내 문서를 RAG 답변 생성용 짧은 컨텍스트로 정리한다."""
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
    """분류기가 실제 검색 evidence를 쓸 수 있도록 각 저장소를 1건만 조회한다."""
    book_probe = book_retriever.retrieve(keyword, limit=1)
    guide_probe = guide_retriever.retrieve(keyword, limit=1)
    return RetrievalEvidence(
        book_hits=len(book_probe.books),
        guide_hits=len(guide_probe.docs),
        book_keyword=keyword,
        guide_keyword=keyword,
    )
