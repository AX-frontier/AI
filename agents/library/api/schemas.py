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
    queryUid: str
    traceId: str
    conversationUid: str
    message: str = Field(min_length=1)


class MatchedBook(BaseModel):
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
    id: int
    title: str
    sourceUrl: str | None = None
    updatedAt: str | None = None


class LibraryChatResponse(BaseModel):
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

