from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FindingSeverity = Literal["HIGH", "MEDIUM", "LOW"]
ReviewStatus = Literal["SUITABLE", "REVISION_REQUIRED", "CHECK_REQUIRED", "NOT_APPLICABLE"]


@dataclass(frozen=True, slots=True)
class DocumentLine:
    number: int
    text: str


@dataclass(frozen=True, slots=True)
class RuleFinding:
    rule_code: str
    category: str
    severity: FindingSeverity
    status: ReviewStatus
    line_start: int
    line_end: int
    original_text: str
    reason: str
    suggested_text: str | None = None
    editable: bool = True


@dataclass(frozen=True, slots=True)
class CheckRequiredItem:
    category: str
    message: str
    original_text: str | None = None
    line_start: int | None = None


@dataclass(frozen=True, slots=True)
class FormatNoticeItem:
    category: str
    message: str
