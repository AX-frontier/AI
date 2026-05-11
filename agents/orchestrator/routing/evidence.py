from __future__ import annotations

from dataclasses import dataclass

from agents.document_review.routing import collect_document_review_evidence
from agents.library.classifier import classify_intent, extract_search_keyword
from agents.library.repository import LibraryRepository, get_library_repository
from agents.library.retrieval import BookRetriever, GuideRetriever, collect_retrieval_evidence
from agents.main_agent.embedding import EmbeddingProvider, get_embedding_provider
from agents.main_agent.repository import MainChunkRepository, get_main_chunk_repository
from agents.main_agent.retrieval import MainRetriever


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
        result = MainRetriever(repository, self._embedding_provider).retrieve(message, limit=3)
        chunks = result.chunks
        if not chunks:
            return AgentEvidence(score=0.0, reason="no main vector chunk hit")

        top = chunks[0]
        hits = ", ".join(
            f"{chunk.title} ({chunk.chunk_id}, {chunk.score:.3f})" for chunk in chunks[:3]
        )
        return AgentEvidence(
            score=round(max(0.0, min(1.0, top.score)), 3),
            reason=f"main reranked hits: {hits}",
        )

    def _collect_library_evidence(self, message: str) -> AgentEvidence:
        repository = self._library_repository or get_library_repository()
        book_retriever = BookRetriever(repository)
        guide_retriever = GuideRetriever(repository)
        initial_classification = classify_intent(message)
        keyword = extract_search_keyword(message, initial_classification.intent)
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
        evidence = collect_document_review_evidence(message)
        return AgentEvidence(score=evidence.score, reason=evidence.reason)
