from __future__ import annotations

import re
from dataclasses import dataclass

from agents.library.classifier import classify_intent, extract_search_keyword
from agents.library.repository import LibraryRepository, get_library_repository
from agents.library.retrieval import BookRetriever, GuideRetriever, collect_retrieval_evidence
from agents.main_agent.embedding import EmbeddingProvider, get_embedding_provider
from agents.main_agent.repository import MainChunkRepository, get_main_chunk_repository
from agents.main_agent.retrieval import extract_main_search_keyword


@dataclass(frozen=True)
class AgentEvidence:
    score: float
    reason: str


@dataclass(frozen=True)
class RoutingEvidence:
    main: AgentEvidence
    library: AgentEvidence
    document_review: AgentEvidence


class RoutingEvidenceCollector:
    """Main/Library/Document Review 처리 가능성을 가볍게 수집한다."""

    def __init__(
        self,
        *,
        main_repository: MainChunkRepository | None = None,
        library_repository: LibraryRepository | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self._main_repository = main_repository
        self._library_repository = library_repository
        self._embedding_provider = embedding_provider or get_embedding_provider()

    def collect(self, message: str) -> RoutingEvidence:
        text = message.strip()
        return RoutingEvidence(
            main=self._collect_main_evidence(text),
            library=self._collect_library_evidence(text),
            document_review=self._collect_document_review_evidence(text),
        )

    def _collect_main_evidence(self, message: str) -> AgentEvidence:
        repository = self._main_repository or get_main_chunk_repository()
        keyword = extract_main_search_keyword(message)
        embedding = self._embedding_provider.embed_query(keyword)
        chunks = repository.search_similar_chunks(embedding, limit=1)
        if not chunks:
            return AgentEvidence(score=0.0, reason="no main vector chunk hit")

        top = chunks[0]
        title = top.title
        return AgentEvidence(
            score=round(max(0.0, min(1.0, top.score)), 3),
            reason=f"top main chunk: {title} ({top.chunk_id})",
        )

    def _collect_library_evidence(self, message: str) -> AgentEvidence:
        repository = self._library_repository or get_library_repository()
        book_retriever = BookRetriever(repository)
        guide_retriever = GuideRetriever(repository)
        keyword = extract_search_keyword(message)
        retrieval_evidence = collect_retrieval_evidence(
            keyword,
            book_retriever=book_retriever,
            guide_retriever=guide_retriever,
        )
        classification = classify_intent(message, evidence=retrieval_evidence)
        return AgentEvidence(
            score=classification.confidence,
            reason=classification.reason,
        )

    def _collect_document_review_evidence(self, message: str) -> AgentEvidence:
        # TODO(dev-B integration): Document Review Agent의 route/classifier API가
        # 확정되면 이 임시 키워드 기반 evidence를 제거하고 B API 응답의
        # confidence/reason을 사용한다. 현재는 B 담당 구현이 아직 없어서
        # v0.2 라우팅 테스트용으로만 사용한다.
        normalized = message.lower()
        matched = [keyword for keyword in DOCUMENT_REVIEW_KEYWORDS if keyword in normalized]
        if not matched:
            return AgentEvidence(score=0.0, reason="no document review keyword")

        # Temporary heuristic score, not model-based confidence:
        # 1 keyword ~= 0.60, 2 keywords ~= 0.75, capped at 0.95.
        score = min(0.95, 0.45 + (0.15 * len(matched)))
        if _looks_like_review_command(normalized):
            score = max(score, 0.75)
        return AgentEvidence(
            score=round(score, 3),
            reason=f"matched document review keywords: {', '.join(matched)}",
        )


# Temporary fallback keywords for Document Review routing.
# Replace with Document Review Agent evidence API when developer B endpoint is ready.
DOCUMENT_REVIEW_KEYWORDS = (
    "문서",
    "전자결재",
    "공문",
    "검토",
    "수정",
    "두문",
    "본문",
    "결문",
    "맞춤법",
    "문장",
)


def _looks_like_review_command(message: str) -> bool:
    return bool(re.search(r"(검토|수정|확인|고쳐|봐줘|봐 줘)", message))
