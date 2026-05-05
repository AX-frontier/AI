from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LibraryIntent = Literal[
    "BOOK_SEARCH",
    "BOOK_LOCATION",
    "LIBRARY_GUIDE",
    "BOOK_RECOMMENDATION",
    "LIBRARY_GENERAL",
]


class LibraryChatRequest(BaseModel):
    """Spring이 Library Agent로 메시지를 라우팅할 때 보내는 요청 본문."""

    queryUid: str
    traceId: str
    conversationUid: str
    message: str = Field(min_length=1)


class MatchedBook(BaseModel):
    """Spring 호환 camelCase JSON으로 노출되는 도서 결과 필드."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    bibNo: str
    regNo: str
    title: str
    author: str | None = None
    publisher: str | None = None
    publishYear: int | None = None
    holdingCallNo: str | None = None
    materialType: str | None = None
    locationSymbol: str | None = None
    stackLocation: str | None = None
    stackShelf: str | None = None


class LibrarySource(BaseModel):
    """Library 답변에 사용된 안내 문서의 출처 메타데이터."""

    id: int
    title: str
    sourceUrl: str | None = None
    updatedAt: str | None = None


class LibraryChatResponse(BaseModel):
    """Spring OrchestrateResponse와 호환되는 Library Agent 응답 payload."""

    targetAgent: Literal["LIBRARY"] = "LIBRARY"
    intent: LibraryIntent
    answer: str
    sources: list[LibrarySource | dict[str, Any]] = Field(default_factory=list)
    confidence: float
    fallbackUsed: bool
    fallbackReason: str | None = None
    searchKeyword: str | None = None
    resultCount: int = 0
    matchedBooks: list[MatchedBook] = Field(default_factory=list)
