from __future__ import annotations

from agents.document_review.api.schemas import DocumentReviewRequest, DocumentReviewResponse
from agents.document_review.generator import build_document_review_response, build_fallback_response
from agents.document_review.llm_review import review_with_manual_prompts
from agents.document_review.rules import (
    apply_safe_suggestions,
    apply_safe_suggestions_to_html,
    review_rules,
    review_table_checks,
)
from agents.document_review.tables import extract_tables_from_html
from agents.main_agent.llm.base import LLMClient


def run_document_review_agent(
    request: DocumentReviewRequest,
    *,
    llm_client: LLMClient | None = None,
) -> DocumentReviewResponse:
    """전자결재 문서 검토 파이프라인을 실행한다."""
    body_text, used_document_body = _extract_body_text(request)
    if not body_text or len(body_text.strip()) < 5:
        return build_fallback_response("document.bodyText 또는 message에 검토할 본문을 포함해 주세요.")

    extracted_tables = extract_tables_from_html(request.document.bodyHtml if request.document else None)
    findings, checks, format_notices = review_rules(
        body_text,
        extracted_tables,
        source_has_html=bool(request.document and request.document.bodyHtml),
    )
    table_checks_available = True
    try:
        table_checks = review_table_checks(body_text, extracted_tables)
    except Exception:
        table_checks = []
        table_checks_available = False
    revised_text = apply_safe_suggestions(body_text, findings)
    checks.extend(
        review_with_manual_prompts(
            body_text=body_text,
            revised_text=revised_text,
            findings=findings,
            checks=checks,
            format_notices=format_notices,
            table_checks=table_checks,
            extracted_tables=extracted_tables,
            llm_client=llm_client,
        )
    )
    revised_html = apply_safe_suggestions_to_html(
        request.document.bodyHtml if request.document else None,
        findings,
    )
    return build_document_review_response(
        findings=findings,
        checks=checks,
        format_notices=format_notices,
        table_checks=table_checks,
        table_checks_available=table_checks_available,
        extracted_tables=extracted_tables,
        revised_text=revised_text,
        revised_html=revised_html,
        used_document_body=used_document_body,
    )


def _extract_body_text(request: DocumentReviewRequest) -> tuple[str, bool]:
    if request.document and request.document.bodyText.strip():
        return request.document.bodyText, True
    return request.message, False
