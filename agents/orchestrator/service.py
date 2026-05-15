from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterator

from agents.document_review.agent import run_document_review_agent
from agents.document_review.api.schemas import DocumentReviewRequest
from agents.library.agent import run_library_agent, run_library_agent_stream
from agents.library.api.schemas import LibraryChatRequest
from agents.library.repository import LibraryRepository
from agents.main_agent.agent import run_main_agent
from agents.main_agent.api.schemas import MainChatRequest
from agents.main_agent.embedding import EmbeddingProvider
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.repository import MainChunkRepository
from agents.orchestrator.api.schemas import (
    DocumentReviewInputRequiredResponse,
    OrchestratorChatRequest,
    OrchestratorChatResponse,
    OrchestratorFallbackResponse,
    OrchestratorRouteRequest,
    OrchestratorRouteResponse,
    RoutingEvidencePayload,
)
from agents.main_agent.embedding import get_embedding_provider
from agents.main_agent.llm.factory import get_llm_client
from agents.main_agent.repository import get_main_chunk_repository
from agents.orchestrator.routing.evidence import RoutingEvidenceCollector
from agents.orchestrator.routing.router import EvidenceBasedRouter

logger = logging.getLogger(__name__)
_ORCH_FOLLOWUP_MEMORY: dict[str, dict[str, str]] = {}
_ORCH_FOLLOWUP_TTL = timedelta(minutes=15)
_FOLLOWUP_STICKY_MARGIN = 0.10


@dataclass(frozen=True)
class ResolvedRoute:
    route: OrchestratorRouteResponse
    routing_mode: str
    routing_reason_code: str


def route_query(
    request: OrchestratorRouteRequest,
    *,
    evidence_collector: RoutingEvidenceCollector | None = None,
    router: EvidenceBasedRouter | None = None,
) -> OrchestratorRouteResponse:
    """질의를 evidence 기반으로 라우팅하고 Spring 호환 응답을 반환한다."""
    resolved = resolve_target_agent(
        request.conversationUid,
        request.message,
        evidence_collector=evidence_collector,
        router=router,
    )
    return _build_route_response(request, resolved)


def resolve_target_agent(
    conversation_uid: str,
    message: str,
    *,
    evidence_collector: RoutingEvidenceCollector | None = None,
    router: EvidenceBasedRouter | None = None,
) -> ResolvedRoute:
    started_at = datetime.now(UTC)
    collector = evidence_collector or RoutingEvidenceCollector()
    route_router = router or EvidenceBasedRouter()

    evidence = collector.collect(message)
    fresh_decision = route_router.route(evidence)
    fresh_response = OrchestratorRouteResponse(
        queryUid="",
        traceId="",
        conversationUid=conversation_uid,
        targetAgent=fresh_decision.target_agent,
        intent=fresh_decision.intent,
        confidence=round(fresh_decision.confidence, 3),
        reason=fresh_decision.reason,
        evidence=RoutingEvidencePayload(
            mainScore=evidence.main.score,
            libraryScore=evidence.library.score,
            documentReviewScore=evidence.document_review.score,
            mainReason=evidence.main.reason,
            libraryReason=evidence.library.reason,
            documentReviewReason=evidence.document_review.reason,
        ),
        routingMode="FRESH",
        routingReasonCode="FRESH_DEFAULT",
    )

    override = _parse_user_agent_override(message)
    if override is not None:
        overridden = fresh_response.model_copy(
            update={
                "targetAgent": override,
                "intent": override,
                "routingMode": "FRESH",
                "routingReasonCode": "USER_OVERRIDE",
                "reason": f"user explicit override to {override}",
                "confidence": max(0.9, fresh_response.confidence),
            }
        )
        logger.info(
            "orchestrator.route resolved conversation_uid=%s target_agent=%s mode=%s reason_code=%s confidence=%.3f elapsed_ms=%d",
            conversation_uid,
            overridden.targetAgent,
            "FRESH",
            "USER_OVERRIDE",
            overridden.confidence,
            int((datetime.now(UTC) - started_at).total_seconds() * 1000),
        )
        return ResolvedRoute(overridden, "FRESH", "USER_OVERRIDE")

    memory = _get_valid_followup_memory(conversation_uid)
    if not _looks_followup(message) or memory is None:
        logger.info(
            "orchestrator.route resolved conversation_uid=%s target_agent=%s mode=%s reason_code=%s confidence=%.3f elapsed_ms=%d",
            conversation_uid,
            fresh_response.targetAgent,
            "FRESH",
            "FRESH_DEFAULT",
            fresh_response.confidence,
            int((datetime.now(UTC) - started_at).total_seconds() * 1000),
        )
        return ResolvedRoute(fresh_response, "FRESH", "FRESH_DEFAULT")

    previous_target = memory.get("targetAgent")
    top_target, top_score, second_score = _top_two_scores(evidence)
    if previous_target and previous_target != top_target:
        if top_score - second_score < _FOLLOWUP_STICKY_MARGIN and _agent_score(evidence, previous_target) > 0:
            sticky = fresh_response.model_copy(
                update={
                    "targetAgent": previous_target,
                    "intent": previous_target,
                    "confidence": round(_agent_score(evidence, previous_target), 3),
                    "reason": "follow-up reuse selected by tie-break",
                    "routingMode": "FOLLOWUP_STICKY",
                    "routingReasonCode": "FOLLOWUP_REUSE",
                }
            )
            logger.info(
                "orchestrator.route resolved conversation_uid=%s target_agent=%s mode=%s reason_code=%s confidence=%.3f elapsed_ms=%d",
                conversation_uid,
                sticky.targetAgent,
                "FOLLOWUP_STICKY",
                "FOLLOWUP_REUSE",
                sticky.confidence,
                int((datetime.now(UTC) - started_at).total_seconds() * 1000),
            )
            return ResolvedRoute(sticky, "FOLLOWUP_STICKY", "FOLLOWUP_REUSE")
        switched = fresh_response.model_copy(
            update={
                "routingMode": "FRESH",
                "routingReasonCode": "EVIDENCE_SWITCH",
            }
        )
        logger.info(
            "orchestrator.route resolved conversation_uid=%s target_agent=%s mode=%s reason_code=%s confidence=%.3f elapsed_ms=%d",
            conversation_uid,
            switched.targetAgent,
            "FRESH",
            "EVIDENCE_SWITCH",
            switched.confidence,
            int((datetime.now(UTC) - started_at).total_seconds() * 1000),
        )
        return ResolvedRoute(switched, "FRESH", "EVIDENCE_SWITCH")

    logger.info(
        "orchestrator.route resolved conversation_uid=%s target_agent=%s mode=%s reason_code=%s confidence=%.3f elapsed_ms=%d",
        conversation_uid,
        fresh_response.targetAgent,
        "FRESH",
        "FRESH_DEFAULT",
        fresh_response.confidence,
        int((datetime.now(UTC) - started_at).total_seconds() * 1000),
    )
    return ResolvedRoute(fresh_response, "FRESH", "FRESH_DEFAULT")


def _top_two_scores(evidence) -> tuple[str, float, float]:
    ranked = sorted(
        [
            ("MAIN", evidence.main.score),
            ("LIBRARY", evidence.library.score),
            ("DOCUMENT_REVIEW", evidence.document_review.score),
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    top_target, top_score = ranked[0]
    second_score = ranked[1][1]
    return top_target, top_score, second_score


def _agent_score(evidence, target: str) -> float:
    if target == "MAIN":
        return evidence.main.score
    if target == "LIBRARY":
        return evidence.library.score
    if target == "DOCUMENT_REVIEW":
        return evidence.document_review.score
    return 0.0


def _parse_user_agent_override(message: str) -> str | None:
    lowered = message.lower()
    if "메인 에이전트" in message or "main agent" in lowered:
        return "MAIN"
    if "도서" in message and "에이전트" in message:
        return "LIBRARY"
    if "라이브러리 에이전트" in message or "library agent" in lowered:
        return "LIBRARY"
    if "문서 검토 에이전트" in message or "document review" in lowered:
        return "DOCUMENT_REVIEW"
    return None


def _get_valid_followup_memory(conversation_uid: str) -> dict[str, str] | None:
    memory = _ORCH_FOLLOWUP_MEMORY.get(conversation_uid)
    if not memory:
        return None
    updated_at = memory.get("updatedAt")
    if not updated_at:
        return memory
    try:
        updated_at_dt = datetime.fromisoformat(updated_at)
    except ValueError:
        return memory
    if datetime.now(UTC) - updated_at_dt > _ORCH_FOLLOWUP_TTL:
        _ORCH_FOLLOWUP_MEMORY.pop(conversation_uid, None)
        return None
    return memory


def _build_route_response(
    request: OrchestratorRouteRequest,
    resolved: ResolvedRoute,
) -> OrchestratorRouteResponse:
    route_response = resolved.route
    return OrchestratorRouteResponse(
        queryUid=request.queryUid,
        traceId=request.traceId,
        conversationUid=request.conversationUid,
        targetAgent=route_response.targetAgent,
        intent=route_response.intent,
        confidence=route_response.confidence,
        reason=route_response.reason,
        evidence=route_response.evidence,
        routingMode=resolved.routing_mode,
        routingReasonCode=resolved.routing_reason_code,
    )


def execute_routed_query(
    request: OrchestratorChatRequest,
    *,
    evidence_collector: RoutingEvidenceCollector | None = None,
    router: EvidenceBasedRouter | None = None,
    main_repository: MainChunkRepository | None = None,
    main_embedding_provider: EmbeddingProvider | None = None,
    main_llm_client: LLMClient | None = None,
    library_repository: LibraryRepository | None = None,
) -> OrchestratorChatResponse:
    """질의를 라우팅한 뒤 선택된 Agent를 내부 함수 호출로 바로 실행한다."""
    started_at = datetime.now(UTC)
    logger.info(
        "orchestrator.chat start query_uid=%s trace_id=%s conversation_uid=%s message=%s",
        request.queryUid,
        request.traceId,
        request.conversationUid,
        request.message.strip().replace("\n", " ")[:200],
    )
    resolved_message = _resolve_orchestrator_followup_message(
        str(request.conversationUid),
        request.message,
    )
    try:
        resolved_route = resolve_target_agent(
            str(request.conversationUid),
            resolved_message,
            evidence_collector=evidence_collector,
            router=router,
        )
        route_result = _build_route_response(request, resolved_route)
        logger.info(
            "orchestrator.chat routed query_uid=%s target_agent=%s intent=%s mode=%s reason_code=%s confidence=%.3f",
            request.queryUid,
            route_result.targetAgent,
            route_result.intent,
            route_result.routingMode,
            route_result.routingReasonCode,
            route_result.confidence,
        )

        if route_result.targetAgent == "MAIN":
            response = run_main_agent(
                MainChatRequest(
                    queryUid=request.queryUid,
                    traceId=request.traceId,
                    conversationUid=request.conversationUid,
                    message=resolved_message,
                ),
                repository=main_repository,
                embedding_provider=main_embedding_provider,
                llm_client=main_llm_client,
            )
            _remember_orchestrator_context(
                str(request.conversationUid),
                route_result.targetAgent,
                _extract_memory_topic(resolved_message),
            )
            logger.info(
                "orchestrator.chat done query_uid=%s target_agent=%s elapsed_ms=%d",
                request.queryUid,
                route_result.targetAgent,
                int((datetime.now(UTC) - started_at).total_seconds() * 1000),
            )
            return response

        if route_result.targetAgent == "LIBRARY":
            response = run_library_agent(
                LibraryChatRequest.model_construct(
                    queryUid=request.queryUid,
                    traceId=request.traceId,
                    conversationUid=request.conversationUid,
                    message=resolved_message,
                ),
                repository=library_repository,
            )
            topic = resolved_message
            if getattr(response, "summary", None) and isinstance(response.summary, dict):
                title = response.summary.get("title")
                if isinstance(title, str) and title.strip():
                    topic = title.strip()
            _remember_orchestrator_context(
                str(request.conversationUid),
                route_result.targetAgent,
                _extract_memory_topic(topic),
            )
            logger.info(
                "orchestrator.chat done query_uid=%s target_agent=%s elapsed_ms=%d",
                request.queryUid,
                route_result.targetAgent,
                int((datetime.now(UTC) - started_at).total_seconds() * 1000),
            )
            return response

        if route_result.targetAgent == "DOCUMENT_REVIEW":
            if request.document is None or not request.document.bodyText.strip():
                response = _build_document_review_input_required_response(route_result.confidence)
                _remember_orchestrator_context(
                    str(request.conversationUid),
                    route_result.targetAgent,
                    _extract_memory_topic(resolved_message),
                )
                logger.info(
                    "orchestrator.chat done query_uid=%s target_agent=%s requires_document_input=true elapsed_ms=%d",
                    request.queryUid,
                    route_result.targetAgent,
                    int((datetime.now(UTC) - started_at).total_seconds() * 1000),
                )
                return response
            response = run_document_review_agent(
                DocumentReviewRequest(
                    queryUid=request.queryUid,
                    traceId=request.traceId,
                    conversationUid=request.conversationUid,
                    message=resolved_message,
                    document=request.document,
                )
            )
            _remember_orchestrator_context(
                str(request.conversationUid),
                route_result.targetAgent,
                _extract_memory_topic(resolved_message),
            )
            logger.info(
                "orchestrator.chat done query_uid=%s target_agent=%s elapsed_ms=%d",
                request.queryUid,
                route_result.targetAgent,
                int((datetime.now(UTC) - started_at).total_seconds() * 1000),
            )
            return response

        logger.info(
            "orchestrator.chat fallback query_uid=%s elapsed_ms=%d",
            request.queryUid,
            int((datetime.now(UTC) - started_at).total_seconds() * 1000),
        )
        return _build_fallback_response("질문을 이해하지 못했습니다. 관련 주제로 다시 입력해 주세요.")
    except Exception:
        logger.exception(
            "orchestrator.chat failed query_uid=%s trace_id=%s conversation_uid=%s",
            request.queryUid,
            request.traceId,
            request.conversationUid,
        )
        raise


def stream_orchestrator_chat(
    request: OrchestratorChatRequest,
    *,
    evidence_collector: RoutingEvidenceCollector | None = None,
    router: EvidenceBasedRouter | None = None,
    main_repository: MainChunkRepository | None = None,
    main_embedding_provider: EmbeddingProvider | None = None,
    main_llm_client: LLMClient | None = None,
    library_repository: LibraryRepository | None = None,
) -> Iterator[str]:
    """라우팅 결과를 먼저 전송하고 MAIN 에이전트는 LLM 응답을 청크 단위로 스트리밍한다."""
    from agents.main_agent.generator import _build_link_description_prompt, _top_source_links, _format_link_guide_answer, _default_link_description, build_main_fallback_response, MAIN_VECTOR_THRESHOLD
    from agents.main_agent.retrieval import MainRetriever

    resolved_message = _resolve_orchestrator_followup_message(
        str(request.conversationUid),
        request.message,
    )
    resolved_route = resolve_target_agent(
        str(request.conversationUid),
        resolved_message,
        evidence_collector=evidence_collector,
        router=router,
    )
    route_result = _build_route_response(request, resolved_route)

    yield f"data: {json.dumps({'type': 'routing', 'targetAgent': route_result.targetAgent, 'intent': route_result.intent, 'routingMode': route_result.routingMode, 'routingReasonCode': route_result.routingReasonCode})}\n\n"

    if route_result.targetAgent == "MAIN":
        repo = main_repository or get_main_chunk_repository()
        embedder = main_embedding_provider or get_embedding_provider()
        llm = main_llm_client or get_llm_client()
        result = MainRetriever(repo, embedder).retrieve(resolved_message.strip())

        if not result.chunks or result.chunks[0].score < MAIN_VECTOR_THRESHOLD:
            fallback = build_main_fallback_response(result.keyword, "관련 공지를 찾지 못했습니다.")
            yield f"data: {json.dumps({'type': 'done', **fallback.model_dump()})}\n\n"
            return

        links = _top_source_links(result.chunks)
        prompt = _build_link_description_prompt(result.keyword, links)

        accumulated = ""
        try:
            for chunk in llm.generate_stream(prompt):
                accumulated += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        except Exception:
            accumulated = "\n".join(_default_link_description(l) for l in links)

        from agents.main_agent.generator import _parse_link_descriptions
        descriptions = _parse_link_descriptions(accumulated, len(links))
        if descriptions is None:
            descriptions = [_default_link_description(l) for l in links]
        final_answer = _format_link_guide_answer(result.keyword, links, descriptions)

        from agents.main_agent.api.schemas import MainSource
        sources = [MainSource(title=c.title, url=c.url, documentId=c.document_id, chunkId=c.chunk_id, category=c.category, postedDate=c.posted_date, score=round(c.score, 3)) for c in result.chunks]
        yield f"data: {json.dumps({'type': 'done', 'targetAgent': 'MAIN', 'intent': 'SCHOOL_NOTICE_QA', 'answer': final_answer, 'sources': [s.model_dump() for s in sources], 'confidence': round(max(0.5, min(0.95, result.chunks[0].score)), 3), 'fallbackUsed': False, 'fallbackReason': None, 'searchKeyword': result.keyword, 'resultCount': len(result.chunks), 'requiresDocumentInput': False})}\n\n"
        return

    if route_result.targetAgent == "LIBRARY":
        llm = main_llm_client or get_llm_client()
        for event in run_library_agent_stream(
            LibraryChatRequest.model_construct(
                queryUid=request.queryUid,
                traceId=request.traceId,
                conversationUid=request.conversationUid,
                message=resolved_message,
            ),
            repository=library_repository,
            llm_client=llm,
        ):
            yield f"data: {json.dumps(event)}\n\n"
        return

    # DOCUMENT_REVIEW 및 나머지 케이스: 동기 실행 후 chunk + done 이벤트로 전송
    response = execute_routed_query(
        request.model_copy(update={"message": resolved_message}),
        evidence_collector=evidence_collector,
        router=router,
        main_repository=main_repository,
        main_embedding_provider=main_embedding_provider,
        main_llm_client=main_llm_client,
        library_repository=library_repository,
    )
    yield f"data: {json.dumps({'type': 'done', **response.model_dump()})}\n\n"


def _resolve_orchestrator_followup_message(conversation_uid: str, message: str) -> str:
    text = (message or "").strip()
    if not _looks_followup(text):
        return text
    memory = _ORCH_FOLLOWUP_MEMORY.get(conversation_uid)
    if not memory:
        return text
    topic = memory.get("topic")
    if not topic:
        return text
    if topic in text:
        return text
    if text in topic:
        return topic
    return f"{topic} {text}"


def _looks_followup(message: str) -> bool:
    lowered = message.lower()
    if len(lowered) > 40:
        return False
    return any(
        token in lowered
        for token in ("그럼", "그러면", "그거", "거긴", "거기", "토요일은", "일요일은", "오늘은", "지금은")
    )


def _extract_memory_topic(message: str) -> str:
    terms = [token for token in message.split() if token and token not in {"그럼", "그러면", "그거", "거긴", "거기"}]
    compact: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        compact.append(term)
        seen.add(term)
    return " ".join(compact[:4]) if compact else message


def _remember_orchestrator_context(conversation_uid: str, target_agent: str, topic: str) -> None:
    _ORCH_FOLLOWUP_MEMORY[conversation_uid] = {
        "targetAgent": target_agent,
        "topic": topic,
        "updatedAt": datetime.now(UTC).isoformat(),
    }
    if len(_ORCH_FOLLOWUP_MEMORY) > 1000:
        oldest = next(iter(_ORCH_FOLLOWUP_MEMORY.keys()))
        _ORCH_FOLLOWUP_MEMORY.pop(oldest, None)


def _build_fallback_response(reason: str) -> OrchestratorFallbackResponse:
    return OrchestratorFallbackResponse(
        answer=reason,
        fallbackReason=reason,
    )


def _build_document_review_input_required_response(confidence: float) -> DocumentReviewInputRequiredResponse:
    return DocumentReviewInputRequiredResponse(confidence=confidence)
