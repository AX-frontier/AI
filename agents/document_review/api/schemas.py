from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReviewDocument(BaseModel):
    title: str | None = None
    docType: str | None = None
    bodyText: str = Field(min_length=1)
    bodyHtml: str | None = None
    editorJson: dict[str, Any] | None = None
    attachmentNames: list[str] = Field(default_factory=list)


class DocumentReviewRequest(BaseModel):
    queryUid: str
    traceId: str
    conversationUid: str
    message: str = Field(default="전자결재 문서를 검토해줘")
    document: ReviewDocument | None = None


class ReviewSummary(BaseModel):
    overallOpinion: str
    reviewScope: str
    totalFindingCount: int
    highCount: int
    mediumCount: int
    lowCount: int


class ReviewFinding(BaseModel):
    id: str
    ruleCode: str
    category: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    status: Literal["SUITABLE", "REVISION_REQUIRED", "CHECK_REQUIRED", "NOT_APPLICABLE"]
    lineStart: int
    lineEnd: int
    originalText: str
    suggestedText: str | None = None
    reason: str
    editable: bool = True


class CriterionResult(BaseModel):
    criterion: str
    status: Literal["SUITABLE", "REVISION_REQUIRED", "CHECK_REQUIRED", "NOT_APPLICABLE"]
    reason: str


class CheckRequiredItemResponse(BaseModel):
    id: str
    category: str
    message: str
    originalText: str | None = None
    lineStart: int | None = None


class FormatNoticeItemResponse(BaseModel):
    category: str
    message: str


class RevisedDocument(BaseModel):
    format: Literal["plain_text"] = "plain_text"
    content: str
    htmlContent: str | None = None


class ExtractedTable(BaseModel):
    index: int
    rowCount: int
    columnCount: int
    rows: list[list[str]]


class TableCheckResponse(BaseModel):
    id: str
    tableIndex: int
    tableTitle: str
    category: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    status: Literal["SUITABLE", "REVISION_REQUIRED", "CHECK_REQUIRED", "NOT_APPLICABLE"]
    message: str
    suggestion: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class DocumentReviewResponse(BaseModel):
    targetAgent: Literal["DOCUMENT_REVIEW"] = "DOCUMENT_REVIEW"
    intent: Literal["DOCUMENT_REVIEW"] = "DOCUMENT_REVIEW"
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["COMPLETED", "FAILED"] = "COMPLETED"
    reviewScope: str
    summary: ReviewSummary
    findings: list[ReviewFinding]
    criterionResults: list[CriterionResult]
    checkRequiredItems: list[CheckRequiredItemResponse]
    formatNoticeItems: list[FormatNoticeItemResponse]
    extractedTables: list[ExtractedTable] = Field(default_factory=list)
    tableChecks: list[TableCheckResponse] = Field(default_factory=list)
    tableChecksAvailable: bool = True
    revisedDocument: RevisedDocument
    reviewMarkdown: str
    confidence: float
    fallbackUsed: bool = False
    fallbackReason: str | None = None
