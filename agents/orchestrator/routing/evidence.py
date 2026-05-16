from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging

from agents.document_review.routing import collect_document_review_evidence
from agents.library.classifier import classify_intent, extract_search_keyword
from agents.library.repository import LibraryRepository, get_library_repository
from agents.library.retrieval import BookRetriever, GuideRetriever, collect_retrieval_evidence
from agents.main_agent.embedding import EmbeddingProvider, get_embedding_provider
from agents.main_agent.repository import MainChunkRepository, get_main_chunk_repository
from agents.main_agent.retrieval import MainRetriever

logger = logging.getLogger(__name__)
_LIBRARY_ROUTABLE_INTENTS = {
    "BOOK_SEARCH",
    "BOOK_LOCATION",
    "BOOK_RECOMMENDATION",
    "LIBRARY_GUIDE",
}
_LIBRARY_EVIDENCE_FLOOR_WITH_HITS = 0.4

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

    # 라우터의 명시적 DOCUMENT_REVIEW 선택 기준과 동일한 값
    _DOCUMENT_REVIEW_FAST_PATH_SCORE = 0.85

    def collect(self, message: str) -> RoutingEvidence:
        text = message.strip()
        doc_evidence = self._collect_document_review_evidence(text)
        if doc_evidence.score >= self._DOCUMENT_REVIEW_FAST_PATH_SCORE:
            result = RoutingEvidence(
                main=AgentEvidence(score=0.0, reason="skipped: document review fast path"),
                library=AgentEvidence(score=0.0, reason="skipped: document review fast path"),
                document_review=doc_evidence,
            )
            logger.info(
                "orchestrator.evidence main=%.3f library=%.3f document_review=%.3f fast_path=true",
                result.main.score,
                result.library.score,
                result.document_review.score,
            )
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_main = pool.submit(self._collect_main_evidence, text)
            future_lib  = pool.submit(self._collect_library_evidence, text)
            main_evidence    = future_main.result()
            library_evidence = future_lib.result()
        result = RoutingEvidence(
            main=main_evidence,
            library=library_evidence,
            document_review=doc_evidence,
        )
        logger.info(
            "orchestrator.evidence main=%.3f library=%.3f document_review=%.3f",
            result.main.score,
            result.library.score,
            result.document_review.score,
        )
        return result

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
        guide_hits = retrieval_evidence.guide_hits
        book_hits = retrieval_evidence.book_hits
        intent = classification.intent
        confidence = classification.confidence

        score = confidence if intent in _LIBRARY_ROUTABLE_INTENTS else 0.0
        if guide_hits > 0 or book_hits > 0:
            score = max(score, _LIBRARY_EVIDENCE_FLOOR_WITH_HITS)

        score = round(max(0.0, min(1.0, score)), 3)
        reason = (
            f"library evidence intent={intent} confidence={confidence:.3f} "
            f"book_hits={book_hits} guide_hits={guide_hits}; {classification.reason}"
        )

        if classification.intent not in _LIBRARY_ROUTABLE_INTENTS:
            return AgentEvidence(score=score, reason=reason)
        return AgentEvidence(score=score, reason=reason)

    def _collect_document_review_evidence(self, message: str) -> AgentEvidence:
        evidence = collect_document_review_evidence(message)
        return AgentEvidence(score=evidence.score, reason=evidence.reason)
