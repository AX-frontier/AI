from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

IngestionSource = Literal["hansung_notice", "hsel_library"]


class IngestionRunRequest(BaseModel):
    """Spring Scheduler가 Python ingestion job을 실행할 때 보내는 요청."""

    sources: list[IngestionSource] = Field(
        default_factory=lambda: ["hansung_notice", "hsel_library"],
        min_length=1,
    )
    sinceDate: date | None = None
    maxPages: int | None = Field(default=None, ge=1)
    maxNoticePages: int | None = Field(default=None, ge=1)
    hansungCategories: list[str] | None = None
    crawlPages: bool = True
    crawlNotices: bool = True
    initSchema: bool = False


class CrawlStats(BaseModel):
    savedCount: int = 0
    updatedCount: int = 0
    unchangedCount: int = 0
    skipCount: int = 0
    errorCount: int = 0
    processedPages: int = 0


class VectorStats(BaseModel):
    insertedCount: int = 0
    updatedCount: int = 0
    skippedCount: int = 0
    deletedCount: int = 0
    processedCount: int = 0
    embeddingProvider: str | None = None
    embeddingDimensions: int | None = None


class SourceIngestionResult(BaseModel):
    source: IngestionSource
    status: Literal["COMPLETED", "FAILED"]
    crawl: CrawlStats
    vector: VectorStats
    errorMessage: str | None = None


class IngestionRunResponse(BaseModel):
    jobId: str
    status: Literal["COMPLETED", "PARTIAL_FAILED", "FAILED"]
    startedAt: str
    endedAt: str
    results: list[SourceIngestionResult]


class IngestionJobAcceptedResponse(BaseModel):
    jobId: str
    status: Literal["ACCEPTED"]
    statusUrl: str


class IngestionJobStatusResponse(BaseModel):
    jobId: str
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "PARTIAL_FAILED", "FAILED"]
    startedAt: str | None = None
    endedAt: str | None = None
    results: list[SourceIngestionResult] = Field(default_factory=list)
    errorMessage: str | None = None
