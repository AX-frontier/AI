from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TargetAgent = Literal["MAIN", "LIBRARY", "DOCUMENT_REVIEW", "FALLBACK"]


class OrchestratorRouteRequest(BaseModel):
    """Spring Core가 targetAgent 판단을 위임할 때 보내는 요청 본문."""

    queryUid: str
    traceId: str
    conversationUid: str
    message: str = Field(min_length=1)


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
