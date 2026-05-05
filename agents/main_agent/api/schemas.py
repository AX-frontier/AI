from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MainIntent = Literal[
    "SCHOOL_NOTICE_QA",
    "ACADEMIC_INFO_QA",
    "MAIN_GENERAL",
]


class MainChatRequest(BaseModel):
    """Spring이 Main Agent로 학교 일반 질의를 라우팅할 때 보내는 요청 본문."""

    queryUid: str
    traceId: str
    conversationUid: str
    message: str = Field(min_length=1)


class MainSource(BaseModel):
    """Main Agent 답변에 사용된 학교 공지/안내 chunk 출처."""

    title: str
    url: str | None = None
    documentId: str
    chunkId: str
    category: str | None = None
    postedDate: str | None = None
    score: float | None = None


class MainChatResponse(BaseModel):
    """Spring OrchestrateResponse와 호환되는 Main Agent 응답 payload."""

    targetAgent: Literal["MAIN"] = "MAIN"
    intent: MainIntent
    answer: str
    sources: list[MainSource | dict[str, Any]] = Field(default_factory=list)
    confidence: float
    fallbackUsed: bool
    fallbackReason: str | None = None
    searchKeyword: str | None = None
    resultCount: int = 0
