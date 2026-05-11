from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.document_review.api.schemas import DocumentReviewResponse, ReviewDocument
from agents.library.api.schemas import LibraryChatResponse
from agents.main_agent.api.schemas import MainChatResponse

TargetAgent = Literal["MAIN", "LIBRARY", "DOCUMENT_REVIEW", "FALLBACK"]


class OrchestratorRouteRequest(BaseModel):
    """Spring Core가 targetAgent 판단을 위임할 때 보내는 요청 본문."""

    queryUid: str
    traceId: str
    conversationUid: str
    message: str = Field(min_length=1)


class OrchestratorChatRequest(OrchestratorRouteRequest):
    """라우팅 후 실행까지 위임할 때 사용하는 통합 요청 본문."""

    document: ReviewDocument | None = None


class RoutingEvidencePayload(BaseModel):
    """각 Agent가 질의를 처리할 수 있는지에 대한 evidence 점수."""

    mainScore: float
    libraryScore: float
    documentReviewScore: float
    mainReason: str
    libraryReason: str
    documentReviewReason: str


class OrchestratorRouteResponse(BaseModel):
    """Evidence 기반 라우팅 결과."""

    queryUid: str
    traceId: str
    conversationUid: str
    targetAgent: TargetAgent
    intent: str
    confidence: float
    reason: str
    evidence: RoutingEvidencePayload


class OrchestratorFallbackResponse(BaseModel):
    """실행형 오케스트레이터가 공통 fallback을 반환할 때 쓰는 최소 payload."""

    targetAgent: Literal["FALLBACK"] = "FALLBACK"
    intent: Literal["FALLBACK"] = "FALLBACK"
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    fallbackUsed: bool = True
    fallbackReason: str


class DocumentReviewInputRequiredResponse(BaseModel):
    """문서검토 의도는 확인됐지만 실제 검토 본문 입력이 더 필요한 상태."""

    targetAgent: Literal["DOCUMENT_REVIEW"] = "DOCUMENT_REVIEW"
    intent: Literal["DOCUMENT_REVIEW_REQUIRED"] = "DOCUMENT_REVIEW_REQUIRED"
    answer: str = "검토할 전자결재 문서 본문을 입력해주세요."
    sources: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float
    fallbackUsed: bool = False
    fallbackReason: str | None = None
    requiresDocumentInput: bool = True


OrchestratorChatResponse = (
    MainChatResponse
    | LibraryChatResponse
    | DocumentReviewResponse
    | DocumentReviewInputRequiredResponse
    | OrchestratorFallbackResponse
)
