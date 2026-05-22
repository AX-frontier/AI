from __future__ import annotations

from collections import Counter

from agents.document_review.api.schemas import (
    CheckRequiredItemResponse,
    CriterionResult,
    DocumentReviewResponse,
    ExtractedTable,
    FormatNoticeItemResponse,
    RevisedDocument,
    TableCheckResponse,
    ReviewFinding,
    ReviewSummary,
)
from agents.document_review.models import CheckRequiredItem, FormatNoticeItem, RuleFinding, TableCheckItem

RULE_CRITERION_MAP = {
    "DATE_FORMAT": "날짜 표기",
    "TIME_FORMAT": "시간 표기",
    "TILDE_SPACING": "문장부호와 기호",
    "COLON_SPACING": "문장부호와 기호",
    "ENDING_EXPRESSION": "문장 종결 표현",
    "RESULT_DELIVERY_PHRASE": "문장 종결 표현",
    "END_MARKER": "끝표시",
    "ATTACHMENT_SPACING": "붙임 표시",
    "ATTACHMENT_LABEL": "붙임 표시",
    "ITEM_SPACING": "항목 번호 체계",
    "ITEM_MARKER_STYLE": "항목 번호 체계",
    "ITEM_MARKER_SEQUENCE": "항목 번호 체계",
    "ITEM_INDENTATION": "항목 번호 체계",
    "SINGLE_ITEM_NUMBERING": "항목 번호 체계",
    "DECLARED_AMOUNT_MISMATCH": "금액 표기",
    "BUDGET_TABLE_HEADER": "소요예산 표시",
}

CRITERIA = (
    "문서 목적과 표현",
    "맞춤법과 띄어쓰기",
    "숫자 표기",
    "항목 번호 체계",
    "날짜 표기",
    "시간 표기",
    "금액 표기",
    "문장부호와 기호",
    "문장 종결 표현",
    "관련문서 표시",
    "법령 표시",
    "소요예산 표시",
    "붙임 표시",
    "끝표시",
    "서식",
)


def build_document_review_response(
    *,
    findings: list[RuleFinding],
    checks: list[CheckRequiredItem],
    format_notices: list[FormatNoticeItem],
    table_checks: list[TableCheckItem],
    table_checks_available: bool,
    extracted_tables: list[ExtractedTable],
    revised_text: str,
    revised_html: str | None,
    used_document_body: bool,
) -> DocumentReviewResponse:
    """검토 결과를 Spring 호환 응답과 프론트 작업공간용 상세 결과로 변환한다."""
    response_findings = [
        ReviewFinding(
            id=f"finding-{index:03d}",
            ruleCode=finding.rule_code,
            category=finding.category,
            severity=finding.severity,
            status=finding.status,
            lineStart=finding.line_start,
            lineEnd=finding.line_end,
            originalText=finding.original_text,
            suggestedText=finding.suggested_text,
            reason=finding.reason,
            editable=finding.editable,
        )
        for index, finding in enumerate(findings, start=1)
    ]
    check_items = [
        CheckRequiredItemResponse(
            id=f"check-{index:03d}",
            category=item.category,
            message=item.message,
            originalText=item.original_text,
            lineStart=item.line_start,
        )
        for index, item in enumerate(checks, start=1)
    ]
    notice_items = [
        FormatNoticeItemResponse(category=item.category, message=item.message)
        for item in format_notices
    ]
    table_check_items = [
        TableCheckResponse(
            id=f"table-check-{index:03d}",
            tableIndex=item.table_index,
            tableTitle=item.table_title,
            category=item.category,
            severity=item.severity,
            status=item.status,
            message=item.message,
            suggestion=item.suggestion,
            evidence=item.evidence,
        )
        for index, item in enumerate(table_checks, start=1)
    ]
    summary = _build_summary(findings)
    criteria = _build_criteria(findings, checks, format_notices)
    markdown = _build_review_markdown(
        summary,
        response_findings,
        check_items,
        notice_items,
        table_check_items,
        table_checks_available,
    )
    if not used_document_body:
        markdown = (
            "전용 문서 본문 필드가 없어 message를 임시 본문으로 검토했습니다. "
            "프론트 작업공간에서는 document.bodyText로 원문을 보내야 합니다.\n\n"
            f"{markdown}"
        )

    return DocumentReviewResponse(
        answer=markdown,
        reviewScope="전자결재 프롬프트 문서에 명시된 행정문서 작성 규칙만 검토",
        summary=summary,
        findings=response_findings,
        criterionResults=criteria,
        checkRequiredItems=check_items,
        formatNoticeItems=notice_items,
        extractedTables=extracted_tables,
        tableChecks=table_check_items,
        tableChecksAvailable=table_checks_available,
        revisedDocument=RevisedDocument(content=revised_text, htmlContent=revised_html),
        reviewMarkdown=markdown,
        confidence=0.82 if used_document_body else 0.55,
        fallbackUsed=False,
    )


def build_fallback_response(reason: str) -> DocumentReviewResponse:
    summary = ReviewSummary(
        overallOpinion="검토할 전자결재 본문이 충분하지 않습니다.",
        reviewScope="전자결재 프롬프트 문서에 명시된 행정문서 작성 규칙만 검토",
        totalFindingCount=0,
        highCount=0,
        mediumCount=0,
        lowCount=0,
    )
    markdown = f"검토할 전자결재 본문을 찾지 못했습니다. {reason}"
    return DocumentReviewResponse(
        answer=markdown,
        status="FAILED",
        reviewScope=summary.reviewScope,
        summary=summary,
        findings=[],
        criterionResults=[],
        checkRequiredItems=[],
        formatNoticeItems=[],
        extractedTables=[],
        tableChecks=[],
        tableChecksAvailable=False,
        revisedDocument=RevisedDocument(content=""),
        reviewMarkdown=markdown,
        confidence=0.0,
        fallbackUsed=True,
        fallbackReason=reason,
    )


def _build_summary(findings: list[RuleFinding]) -> ReviewSummary:
    counts = Counter(finding.severity for finding in findings)
    total = len(findings)
    if total == 0:
        opinion = "전자결재 프롬프트 기준에서 자동 수정이 필요한 항목은 발견되지 않았습니다."
    else:
        opinion = f"자동 수정 제안 {total}건을 발견했습니다. 사실관계는 변경하지 않고 형식·표현 규칙만 반영했습니다."
    return ReviewSummary(
        overallOpinion=opinion,
        reviewScope="전자결재 프롬프트 문서에 명시된 행정문서 작성 규칙만 검토",
        totalFindingCount=total,
        highCount=counts["HIGH"],
        mediumCount=counts["MEDIUM"],
        lowCount=counts["LOW"],
    )


def _build_criteria(
    findings: list[RuleFinding],
    checks: list[CheckRequiredItem],
    format_notices: list[FormatNoticeItem],
) -> list[CriterionResult]:
    finding_categories = {RULE_CRITERION_MAP.get(finding.rule_code, finding.category) for finding in findings}
    check_categories = {item.category for item in checks}
    notice_categories = {item.category for item in format_notices}
    results: list[CriterionResult] = []
    for criterion in CRITERIA:
        if criterion in finding_categories:
            results.append(
                CriterionResult(
                    criterion=criterion,
                    status="REVISION_REQUIRED",
                    reason="프롬프트 규칙에 따라 자동 수정 제안이 있습니다.",
                )
            )
        elif criterion in check_categories:
            results.append(
                CriterionResult(
                    criterion=criterion,
                    status="CHECK_REQUIRED",
                    reason="문맥 또는 사실관계 확인이 필요한 항목입니다.",
                )
            )
        elif criterion in notice_categories:
            results.append(
                CriterionResult(
                    criterion=criterion,
                    status="CHECK_REQUIRED",
                    reason="텍스트만으로 자동 판정하기 어려워 전자결재 화면에서 확인해야 합니다.",
                )
            )
        else:
            results.append(
                CriterionResult(
                    criterion=criterion,
                    status="SUITABLE",
                    reason="자동 검토 범위에서는 위반 사항이 발견되지 않았습니다.",
                )
            )
    return results


def _build_review_markdown(
    summary: ReviewSummary,
    findings: list[ReviewFinding],
    checks: list[CheckRequiredItemResponse],
    notices: list[FormatNoticeItemResponse],
    table_checks: list[TableCheckResponse],
    table_checks_available: bool,
) -> str:
    lines = [
        "## 문서 검토 결과",
        "",
        f"- 검토 범위: {summary.reviewScope}",
        f"- 수정 제안: {summary.totalFindingCount}건",
        "",
        "### 자동 수정 제안",
    ]
    if findings:
        for finding in findings:
            lines.extend(
                [
                    f"- {finding.category} / {finding.ruleCode} / {finding.lineStart}행",
                    f"  - 원문: {finding.originalText}",
                    f"  - 수정안: {finding.suggestedText or '확인 필요'}",
                    f"  - 사유: {finding.reason}",
                ]
            )
    else:
        lines.append("- 자동 수정 제안 없음")

    lines.extend(["", "### 직접 확인 필요"])
    if checks or table_checks or not table_checks_available:
        for item in checks:
            location = f"{item.lineStart}행" if item.lineStart else "문서 전체"
            lines.append(f"- {item.category} / {location}: {item.message}")
        if table_checks:
            for item in table_checks:
                lines.append(f"- 표 검토 / 표 {item.tableIndex} {item.tableTitle}: {item.message} 권장 조치: {item.suggestion}")
        if not table_checks_available:
            lines.append("- 표 검토 / 문서 전체: 표 검토를 수행하지 못했습니다. 원본 전자결재/HWP 표에서 금액과 필수 항목을 직접 확인해 주세요.")
    else:
        lines.append("- 직접 확인 항목 없음")

    lines.extend(["", "### 서식 참고"])
    for item in notices:
        lines.append(f"- {item.category}: {item.message}")
    return "\n".join(lines)
