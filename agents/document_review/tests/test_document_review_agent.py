from __future__ import annotations

from agents.document_review.agent import run_document_review_agent
from agents.document_review.api.schemas import DocumentReviewRequest, ReviewDocument


def test_document_review_agent_suggests_prompt_rule_revisions() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000001",
        traceId="00000000-0000-0000-0000-000000000002",
        conversationUid="00000000-0000-0000-0000-000000000003",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            title="예시 공문",
            docType="OFFICIAL_DOCUMENT",
            bodyText=(
                "2026-04-02 09:10:08\n"
                "1.추진 목적\n"
                "원장 : 김갑동\n"
                "기간: 2026. 3. 3.(화) ~ 3. 20.(금)\n"
                "붙임 1. 산출내역서 1부.\n"
                "결재를 재가하여 주시기 바랍니다.끝."
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert response.targetAgent == "DOCUMENT_REVIEW"
    assert response.fallbackUsed is False
    assert response.summary.totalFindingCount >= 6
    rule_codes = {finding.ruleCode for finding in response.findings}
    assert "DATE_FORMAT" in rule_codes
    assert "ITEM_SPACING" in rule_codes
    assert "COLON_SPACING" in rule_codes
    assert "TILDE_SPACING" in rule_codes
    assert "ATTACHMENT_SPACING" in rule_codes
    assert "END_MARKER" in rule_codes
    assert "2026. 4. 2." in response.revisedDocument.content
    assert "1. 추진 목적" in response.revisedDocument.content
    assert "원장: 김갑동" in response.revisedDocument.content
    assert "3. 3.(화)~3. 20.(금)" in response.revisedDocument.content
    assert "붙임  1." in response.revisedDocument.content


def test_document_review_agent_accepts_spring_current_message_only_shape() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000001",
        traceId="00000000-0000-0000-0000-000000000002",
        conversationUid="00000000-0000-0000-0000-000000000003",
        message="2026-05-04 10:26:39\n붙임 1. 안내문 1부.\n끝.",
    )

    response = run_document_review_agent(request)

    assert response.targetAgent == "DOCUMENT_REVIEW"
    assert response.fallbackUsed is False
    assert response.confidence == 0.55
    assert "전용 문서 본문 필드가 없어" in response.answer


def test_document_review_agent_returns_fallback_for_empty_body() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000001",
        traceId="00000000-0000-0000-0000-000000000002",
        conversationUid="00000000-0000-0000-0000-000000000003",
        message="검토",
    )

    response = run_document_review_agent(request)

    assert response.status == "FAILED"
    assert response.fallbackUsed is True


def test_document_review_agent_does_not_flag_matching_korean_amount() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000001",
        traceId="00000000-0000-0000-0000-000000000002",
        conversationUid="00000000-0000-0000-0000-000000000003",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText="정산 금액: 금269,320원(금이십육만구천삼백이십원)\n  끝.",
        ),
    )

    response = run_document_review_agent(request)

    assert all(item.category != "금액 표기" for item in response.checkRequiredItems)


def test_document_review_agent_does_not_infer_budget_table_from_keyword_only() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000001",
        traceId="00000000-0000-0000-0000-000000000002",
        conversationUid="00000000-0000-0000-0000-000000000003",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText="3. 소요예산\n\n본 사업은 별도 예산 없이 진행합니다.\n\n  끝.",
        ),
    )

    response = run_document_review_agent(request)

    assert all(item.category != "소요예산 표시" for item in response.checkRequiredItems)


def test_document_review_agent_extracts_html_tables() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000001",
        traceId="00000000-0000-0000-0000-000000000002",
        conversationUid="00000000-0000-0000-0000-000000000003",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText="소요예산\n회계연도 예산구분 세목 세목코드 소요예산\n  끝.",
            bodyHtml=(
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>세목코드</td><td>소요예산</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>9911001</td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert len(response.extractedTables) == 1
    assert response.extractedTables[0].rows[0] == ["회계연도", "예산구분", "세목", "세목코드", "소요예산"]


def test_document_review_agent_checks_related_document_and_law_reference() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000001",
        traceId="00000000-0000-0000-0000-000000000002",
        conversationUid="00000000-0000-0000-0000-000000000003",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "관련: 총무인사팀 출장허가신청서\n"
                "개인정보 보호법 제21조에 따라 처리합니다.\n"
                "  끝."
            ),
        ),
    )

    response = run_document_review_agent(request)

    categories = {item.category for item in response.checkRequiredItems}
    assert "관련문서 표시" in categories
    assert "법령 표시" in categories


def test_document_review_agent_checks_budget_table_required_columns_from_html() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000001",
        traceId="00000000-0000-0000-0000-000000000002",
        conversationUid="00000000-0000-0000-0000-000000000003",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText="마. 소요예산\n  끝.",
            bodyHtml=(
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>소요예산</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    budget_checks = [item for item in response.checkRequiredItems if item.category == "소요예산 표시"]
    assert budget_checks
    assert "세목코드" in budget_checks[0].message
