from __future__ import annotations

from agents.document_review.agent import run_document_review_agent
from agents.document_review.api.schemas import DocumentReviewRequest, ReviewDocument, TableCheckResponse
from agents.document_review.models import RuleFinding
from agents.document_review.rules import apply_safe_suggestions_to_html, review_rules


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


def test_end_marker_spacing_requires_exact_two_spaces() -> None:
    attached_findings, attached_checks, _ = review_rules("붙임  1. 안내문 1부.끝.")
    one_space_findings, one_space_checks, _ = review_rules("붙임  1. 안내문 1부. 끝.")
    two_space_findings, two_space_checks, _ = review_rules("붙임  1. 안내문 1부.  끝.")
    tab_findings, tab_checks, _ = review_rules("붙임  1. 안내문 1부.\t끝.")
    one_nbsp_findings, one_nbsp_checks, _ = review_rules("붙임  1. 안내문 1부.\u00a0끝.")
    nbsp_space_findings, nbsp_space_checks, _ = review_rules("붙임  1. 안내문 1부.\u00a0 끝.")

    assert any(finding.rule_code == "END_MARKER" for finding in attached_findings)
    assert not any(check.category == "끝표시" for check in attached_checks)
    assert any(finding.rule_code == "END_MARKER" for finding in one_space_findings)
    assert not any(check.category == "끝표시" for check in one_space_checks)
    assert not any(finding.rule_code == "END_MARKER" for finding in two_space_findings)
    assert not any(check.category == "끝표시" for check in two_space_checks)
    assert any(finding.rule_code == "END_MARKER" for finding in tab_findings)
    assert not any(check.category == "끝표시" for check in tab_checks)
    assert any(finding.rule_code == "END_MARKER" for finding in one_nbsp_findings)
    assert not any(check.category == "끝표시" for check in one_nbsp_checks)
    assert not any(finding.rule_code == "END_MARKER" for finding in nbsp_space_findings)
    assert not any(check.category == "끝표시" for check in nbsp_space_checks)


def test_end_marker_requires_sentence_period_before_marker() -> None:
    findings, checks, _ = review_rules("제출 완료  끝.")

    end_marker_findings = [finding for finding in findings if finding.rule_code == "END_MARKER"]

    assert end_marker_findings
    assert end_marker_findings[0].suggested_text == "제출 완료.  끝."
    assert not any(check.category == "끝표시" for check in checks)


def test_weekday_date_requires_day_dot_and_no_space_before_weekday() -> None:
    missing_dot_findings, _, _ = review_rules("일시: 2013. 6. 27(목)")
    spaced_weekday_findings, _, _ = review_rules("일시: 2013. 6. 27. (목)")
    clean_findings, _, _ = review_rules("일시: 2013. 6. 27.(목)")

    assert any(finding.suggested_text == "2013. 6. 27.(목)" for finding in missing_dot_findings)
    assert any(finding.suggested_text == "2013. 6. 27.(목)" for finding in spaced_weekday_findings)
    assert not any(finding.rule_code == "DATE_FORMAT" for finding in clean_findings)


def test_document_review_agent_checks_item_marker_hierarchy() -> None:
    findings, checks, _ = review_rules(
        "1) 학술정보팀 수입을 정산합니다.\n"
        "2. 수입 정산 내용\n"
        " 가. 정산 대상\n"
        " 나. 정산 기간\n"
        "다) 정산 금액\n"
        "  끝."
    )

    hierarchy_checks = [item for item in checks if item.category == "항목 번호 체계"]
    marker_findings = [item for item in findings if item.rule_code == "ITEM_MARKER_STYLE"]

    assert marker_findings
    assert any(item.original_text == "1) 학술정보팀 수입을 정산합니다." and item.suggested_text == "1. 학술정보팀 수입을 정산합니다." for item in marker_findings)
    assert any(item.original_text == "다) 정산 금액" and item.suggested_text == "다. 정산 금액" for item in marker_findings)
    assert any("2타씩 오른쪽" in item.message for item in hierarchy_checks)


def test_document_review_agent_accepts_manual_item_marker_order() -> None:
    _, checks, _ = review_rules(
        "1. 수입 정산 내용\n"
        "  가. 정산 대상\n"
        "  나. 정산 기간\n"
        "    1) 상세 내역\n"
        "    2) 소요예산\n"
        "2. 붙임\n"
        "  끝."
    )

    assert all(item.category != "항목 번호 체계" for item in checks)


def test_document_review_agent_suggests_attachment_label_as_butim() -> None:
    findings, checks, _ = review_rules(
        "첨부  1. 도서 연체료 1부.\n"
        "      2. 문서 출력료 1부.\n"
        "      3. 연회비 1부.  끝."
    )

    label_findings = [item for item in findings if item.rule_code == "ATTACHMENT_LABEL"]

    assert label_findings
    assert label_findings[0].suggested_text == "붙임  1. 도서 연체료 1부."
    assert "첨부" in label_findings[0].original_text
    assert any(item.category == "붙임 표시" for item in checks)


def test_document_review_agent_applies_attachment_label_to_html_content() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000019",
        traceId="00000000-0000-0000-0000-000000000020",
        conversationUid="00000000-0000-0000-0000-000000000021",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText="첨부  1. 도서 연체료 1부.\n      2. 문서 출력료 1부.  끝.",
            bodyHtml="<p>첨부&nbsp;&nbsp;1. 도서 연체료 1부.</p><p>2. 문서 출력료 1부.  끝.</p>",
        ),
    )

    response = run_document_review_agent(request)

    assert "붙임" in response.revisedDocument.content
    assert response.revisedDocument.htmlContent is not None
    assert "붙임" in response.revisedDocument.htmlContent
    assert "첨부" not in response.revisedDocument.htmlContent


def test_document_review_html_auto_fix_does_not_modify_table_cells() -> None:
    revised_html = apply_safe_suggestions_to_html(
        (
            "<table><tr><td>첨부&nbsp;&nbsp;1. 표 안 문구 1부.</td><td>비고</td></tr>"
            "<tr><td>금액</td><td>1,000</td></tr></table>"
            "<p>첨부&nbsp;&nbsp;2. 본문 문구 1부.</p>"
        ),
        [
            RuleFinding(
                rule_code="ATTACHMENT_LABEL",
                category="붙임 표시",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=1,
                line_end=1,
                original_text="첨부  1.",
                suggested_text="붙임  1.",
                reason="첨부파일 목록 표기는 붙임으로 씁니다.",
            )
        ],
    )

    assert revised_html is not None
    assert "<td>첨부" in revised_html
    assert "<p>붙임" in revised_html


def test_document_review_html_auto_fix_allows_outer_layout_table_text() -> None:
    revised_html = apply_safe_suggestions_to_html(
        (
            "<table><tr><td>"
            "<p>첨부&nbsp;&nbsp;1. 본문 문구 1부.</p>"
            "</td></tr></table>"
        ),
        [
            RuleFinding(
                rule_code="ATTACHMENT_LABEL",
                category="붙임 표시",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=1,
                line_end=1,
                original_text="첨부  1.",
                suggested_text="붙임  1.",
                reason="첨부파일 목록 표기는 붙임으로 씁니다.",
            )
        ],
    )

    assert revised_html is not None
    assert "붙임" in revised_html
    assert "첨부" not in revised_html


def test_document_review_html_auto_fix_allows_letterhead_layout_table_text() -> None:
    revised_html = apply_safe_suggestions_to_html(
        (
            "<table>"
            "<tr><td>수신</td><td>학술정보팀</td></tr>"
            "<tr><td>제목</td><td>첨부&nbsp;&nbsp;1. 본문 문구 1부.</td></tr>"
            "</table>"
        ),
        [
            RuleFinding(
                rule_code="ATTACHMENT_LABEL",
                category="붙임 표시",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=1,
                line_end=1,
                original_text="첨부  1.",
                suggested_text="붙임  1.",
                reason="첨부파일 목록 표기는 붙임으로 씁니다.",
            )
        ],
    )

    assert revised_html is not None
    assert "붙임" in revised_html
    assert "첨부" not in revised_html


def test_document_review_html_auto_fix_skips_nested_data_table_text() -> None:
    revised_html = apply_safe_suggestions_to_html(
        (
            "<table><tr><td>"
            "<p>첨부&nbsp;&nbsp;1. 본문 문구 1부.</p>"
            "<table><tr><td>첨부&nbsp;&nbsp;2. 표 안 문구 1부.</td><td>비고</td></tr>"
            "<tr><td>금액</td><td>1,000</td></tr></table>"
            "</td></tr></table>"
        ),
        [
            RuleFinding(
                rule_code="ATTACHMENT_LABEL",
                category="붙임 표시",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=1,
                line_end=1,
                original_text="첨부  1.",
                suggested_text="붙임  1.",
                reason="첨부파일 목록 표기는 붙임으로 씁니다.",
            ),
            RuleFinding(
                rule_code="ATTACHMENT_LABEL",
                category="붙임 표시",
                severity="LOW",
                status="REVISION_REQUIRED",
                line_start=2,
                line_end=2,
                original_text="첨부  2.",
                suggested_text="붙임  2.",
                reason="첨부파일 목록 표기는 붙임으로 씁니다.",
            ),
        ],
    )

    assert revised_html is not None
    assert "붙임" in revised_html
    assert "<td>첨부" in revised_html


def test_document_review_agent_applies_item_marker_style_to_revised_content() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000022",
        traceId="00000000-0000-0000-0000-000000000023",
        conversationUid="00000000-0000-0000-0000-000000000024",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "1) 학술정보팀 수입(2026년 4월분)을 정산합니다.\n"
                "2. 수입 정산 내용\n"
                " 가. 정산 대상\n"
                " 다) 정산 금액\n"
                "붙임  1. 도서 연체료 1부.  끝."
            ),
            bodyHtml=(
                "<p>1) 학술정보팀 수입(2026년 4월분)을 정산합니다.</p>"
                "<p>2. 수입 정산 내용</p>"
                "<p>&nbsp;가. 정산 대상</p>"
                "<p>&nbsp;다) 정산 금액</p>"
                "<p>붙임&nbsp;&nbsp;1. 도서 연체료 1부.  끝.</p>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert "1. 학술정보팀 수입" in response.revisedDocument.content
    assert "다. 정산 금액" in response.revisedDocument.content
    assert "1) 학술정보팀 수입" not in response.revisedDocument.content
    assert "다) 정산 금액" not in response.revisedDocument.content
    assert response.revisedDocument.htmlContent is not None
    assert "1. 학술정보팀 수입" in response.revisedDocument.htmlContent
    assert "다. 정산 금액" in response.revisedDocument.htmlContent


def test_document_review_agent_checks_manual_basic_principles() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000025",
        traceId="00000000-0000-0000-0000-000000000026",
        conversationUid="00000000-0000-0000-0000-000000000027",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "SW TFT 결과를 다음과 같이 보고합니다.\n"
                "참석대상은 열 명입니다.\n"
                "본 문장은 문서의 내용을 둘 이상의 항목으로 구분할 필요가 있음에도 불구하고 매우 긴 문장으로 작성되어 이용자가 핵심 내용을 빠르게 파악하기 어렵고 여러 의미가 한 문장에 함께 포함되어 있어 검토가 필요하며, 문서 작성자가 핵심 목적과 처리 사항을 한눈에 확인하기 어렵게 만들 수 있으므로 표현을 다듬을 여지가 있습니다.\n"
                "  끝."
            ),
        ),
    )

    response = run_document_review_agent(request)

    categories = {item.category for item in response.checkRequiredItems}
    assert "문서 목적과 표현" in categories
    assert "숫자 표기" in categories


def test_document_review_agent_does_not_flag_common_korean_words_as_numbers() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000031",
        traceId="00000000-0000-0000-0000-000000000032",
        conversationUid="00000000-0000-0000-0000-000000000033",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "일부 부서에서 자료를 제출했습니다.\n"
                "이부서 명칭은 예시 문장입니다.\n"
                "참석대상은 10명입니다.\n"
                "  끝."
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert not any(item.category == "숫자 표기" for item in response.checkRequiredItems)


def test_document_review_agent_flags_native_korean_number_with_unit() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000034",
        traceId="00000000-0000-0000-0000-000000000035",
        conversationUid="00000000-0000-0000-0000-000000000036",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "참석 대상은 두 건입니다.\n"
                "  끝."
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert any(item.category == "숫자 표기" for item in response.checkRequiredItems)


def test_document_review_agent_checks_attachment_list_completion() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000028",
        traceId="00000000-0000-0000-0000-000000000029",
        conversationUid="00000000-0000-0000-0000-000000000030",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "붙임  1. 계획서\n"
                "붙임  2. 증빙자료 1부."
            ),
        ),
    )

    response = run_document_review_agent(request)

    messages = [item.message for item in response.checkRequiredItems]
    assert any("부수를 적고 마침표" in message for message in messages)
    assert any("두 번째 붙임부터" in message for message in messages)
    assert any("마지막 붙임 항목 뒤에 끝표시" in message for message in messages)
    assert any("실제 첨부파일명" in message for message in messages)


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
                "<tr><td>합계</td><td></td><td></td><td></td><td>254,400</td></tr>"
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


def test_document_review_agent_returns_table_checks_for_amount_mismatch() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000001",
        traceId="00000000-0000-0000-0000-000000000002",
        conversationUid="00000000-0000-0000-0000-000000000003",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "학술정보팀 수입(2026년 4월분) 정산\n"
                "다. 정산 금액: 금269,320원(금이십육만구천삼백이십원)\n"
                "라. 상세 내역\n"
                "마. 소요예산\n"
                "  끝."
            ),
            bodyHtml=(
                "<table>"
                "<tr><td>구  분</td><td>건 수</td><td>금 액(원)</td><td></td><td>비  고</td></tr>"
                "<tr><td>도서 연체료</td><td>180</td><td>192,400</td><td>192,400</td><td>1권 1일당 100원</td></tr>"
                "<tr><td>문서 출력료</td><td>5</td><td>50,000</td><td>60,000</td><td></td></tr>"
                "<tr><td>연회비</td><td>2</td><td>2,000</td><td>2,000</td><td>1년 10,000원</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td>254,400</td><td></td></tr>"
                "</table>"
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>세목코드</td><td>소요예산</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>9911001</td><td>254,400</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td></td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert response.tableChecks
    assert response.tableChecksAvailable is True
    messages = [item.message for item in response.tableChecks]
    assert any("본문 정산 금액 269,320원과 상세 내역 합계 254,400원이 14,920원 차이" in message for message in messages)
    assert any("본문 정산 금액 269,320원과 소요예산 254,400원이 14,920원 차이" in message for message in messages)


def test_document_review_agent_skips_clean_table_checks() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000004",
        traceId="00000000-0000-0000-0000-000000000005",
        conversationUid="00000000-0000-0000-0000-000000000006",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "학술정보팀 수입 정산\n"
                "다. 정산 금액: 금254,400원(금이십오만사천사백원)\n"
                "라. 상세 내역\n"
                "마. 소요예산\n"
                "  끝."
            ),
            bodyHtml=(
                "<table>"
                "<tr><td>구분</td><td>건수</td><td>금액(원)</td><td></td><td>비고</td></tr>"
                "<tr><td>도서 연체료</td><td>180</td><td>192,400</td><td>192,400</td><td>1권 1일당 100원</td></tr>"
                "<tr><td>문서 출력료</td><td>5</td><td>50,000</td><td>60,000</td><td></td></tr>"
                "<tr><td>연회비</td><td>2</td><td>2,000</td><td>2,000</td><td>1년 10,000원</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td>254,400</td><td></td></tr>"
                "</table>"
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>세목코드</td><td>소요예산</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>9911001</td><td>254,400</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td></td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert response.tableChecks == []


def test_document_review_agent_sums_detail_amount_column_aligned_with_total() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000031",
        traceId="00000000-0000-0000-0000-000000000032",
        conversationUid="00000000-0000-0000-0000-000000000033",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "학술정보팀 수입 정산\n"
                "다. 정산 금액: 금254,400원\n"
                "라. 상세 내역\n"
                "마. 소요예산\n"
                "  끝."
            ),
            bodyHtml=(
                "<table>"
                "<tr><td>구분</td><td>건수</td><td>금액(원)</td><td></td><td>비고</td></tr>"
                "<tr><td>도서 연체료</td><td>180</td><td>192,400</td><td>192,400</td><td>1권 1일당 100원</td></tr>"
                "<tr><td>문서 출력료</td><td>5</td><td>50,000</td><td>60,000</td><td></td></tr>"
                "<tr><td>연회비</td><td>2</td><td>2,000</td><td>2,000</td><td>1년 10,000원</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td>254,400</td><td></td></tr>"
                "</table>"
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>세목코드</td><td>소요예산</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>9911001</td><td>254,400</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td></td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert response.tableChecks == []


def test_document_review_agent_suggests_declared_amount_from_trusted_table_total() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000037",
        traceId="00000000-0000-0000-0000-000000000038",
        conversationUid="00000000-0000-0000-0000-000000000039",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "학술정보팀 수입 정산\n"
                "다. 정산 금액: 금169,320원(금이십육만구천삼백이십원)\n"
                "라. 상세 내역\n"
                "마. 소요예산\n"
                "  끝."
            ),
            bodyHtml=(
                "<p>다. 정산 금액: 금169,320원(금이십육만구천삼백이십원)</p>"
                "<table>"
                "<tr><td>구분</td><td>건수</td><td>금액(원)</td><td></td><td>비고</td></tr>"
                "<tr><td>도서 연체료</td><td>180</td><td>192,400</td><td>192,400</td><td>1권 1일당 100원</td></tr>"
                "<tr><td>문서 출력료</td><td>5</td><td>50,000</td><td>60,000</td><td></td></tr>"
                "<tr><td>연회비</td><td>2</td><td>2,000</td><td>2,000</td><td>1년 10,000원</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td>254,400</td><td></td></tr>"
                "</table>"
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>세목코드</td><td>소요예산</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>9911001</td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert any(finding.ruleCode == "DECLARED_AMOUNT_MISMATCH" for finding in response.findings)
    assert "금254,400원(금이십오만사천사백원)" in response.revisedDocument.content
    assert "금254,400원(금이십오만사천사백원)" in (response.revisedDocument.htmlContent or "")


def test_document_review_agent_applies_declared_amount_when_html_text_is_split() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000041",
        traceId="00000000-0000-0000-0000-000000000042",
        conversationUid="00000000-0000-0000-0000-000000000043",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "학술정보팀 수입 정산\n"
                "다. 정산 금액: 금169,320원(금이십육만구천삼백이십원)\n"
                "라. 상세 내역\n"
                "  끝."
            ),
            bodyHtml=(
                "<p>다. 정산 금액: 금<span>169,320</span>원"
                "<span>(금이십육만구천삼백이십원)</span></p>"
                "<table>"
                "<tr><td>구분</td><td>건수</td><td>금액(원)</td><td></td><td>비고</td></tr>"
                "<tr><td>도서 연체료</td><td>180</td><td>192,400</td><td>192,400</td><td>1권 1일당 100원</td></tr>"
                "<tr><td>문서 출력료</td><td>5</td><td>50,000</td><td>60,000</td><td></td></tr>"
                "<tr><td>연회비</td><td>2</td><td>2,000</td><td>2,000</td><td>1년 10,000원</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td>254,400</td><td></td></tr>"
                "</table>"
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>세목코드</td><td>소요예산</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>9911001</td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert "금254,400원(금이십오만사천사백원)" in (response.revisedDocument.htmlContent or "")
    assert "금169,320원" not in (response.revisedDocument.htmlContent or "")


def test_document_review_agent_does_not_auto_rewrite_declared_amount_without_budget_confirmation() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000044",
        traceId="00000000-0000-0000-0000-000000000045",
        conversationUid="00000000-0000-0000-0000-000000000046",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "학술정보팀 수입 정산\n"
                "다. 정산 금액: 금169,320원(금이십육만구천삼백이십원)\n"
                "라. 상세 내역\n"
                "  끝."
            ),
            bodyHtml=(
                "<p>다. 정산 금액: 금169,320원(금이십육만구천삼백이십원)</p>"
                "<table>"
                "<tr><td>구분</td><td>건수</td><td>금액(원)</td><td></td><td>비고</td></tr>"
                "<tr><td>도서 연체료</td><td>180</td><td>192,400</td><td>192,400</td><td>1권 1일당 100원</td></tr>"
                "<tr><td>문서 출력료</td><td>5</td><td>50,000</td><td>60,000</td><td></td></tr>"
                "<tr><td>연회비</td><td>2</td><td>2,000</td><td>2,000</td><td>1년 10,000원</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td>254,400</td><td></td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert all(finding.ruleCode != "DECLARED_AMOUNT_MISMATCH" for finding in response.findings)
    assert "금169,320원" in response.revisedDocument.content
    assert "금169,320원" in (response.revisedDocument.htmlContent or "")


def test_document_review_agent_declared_amount_html_rewrite_skips_protected_table_text() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000047",
        traceId="00000000-0000-0000-0000-000000000048",
        conversationUid="00000000-0000-0000-0000-000000000049",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "학술정보팀 수입 정산\n"
                "다. 정산 금액: 금169,320원(금이십육만구천삼백이십원)\n"
                "라. 상세 내역\n"
                "마. 소요예산\n"
                "  끝."
            ),
            bodyHtml=(
                "<p>다. 정산 금액: 금<span>169,320</span>원<span>(금이십육만구천삼백이십원)</span></p>"
                "<table>"
                "<tr><td>구분</td><td>건수</td><td>금액(원)</td><td></td><td>비고</td></tr>"
                "<tr><td>도서 연체료</td><td>180</td><td>192,400</td><td>192,400</td><td>금169,320원</td></tr>"
                "<tr><td>문서 출력료</td><td>5</td><td>50,000</td><td>60,000</td><td></td></tr>"
                "<tr><td>연회비</td><td>2</td><td>2,000</td><td>2,000</td><td>1년 10,000원</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td>254,400</td><td></td></tr>"
                "</table>"
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>세목코드</td><td>소요예산</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>9911001</td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    html = response.revisedDocument.htmlContent or ""
    assert "금254,400원(금이십오만사천사백원)" in html
    assert "<td>금169,320원</td>" in html


def test_document_review_agent_flags_budget_header_without_treating_table_as_missing() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000034",
        traceId="00000000-0000-0000-0000-000000000035",
        conversationUid="00000000-0000-0000-0000-000000000036",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "학술정보팀 수입 정산\n"
                "다. 정산 금액: 금254,400원\n"
                "라. 상세 내역\n"
                "마. 소요예산\n"
                "  끝."
            ),
            bodyHtml=(
                "<table>"
                "<tr><td>구분</td><td>건수</td><td>금액(원)</td><td></td><td>비고</td></tr>"
                "<tr><td>도서 연체료</td><td>180</td><td>192,400</td><td>192,400</td><td>1권 1일당 100원</td></tr>"
                "<tr><td>문서 출력료</td><td>5</td><td>50,000</td><td>60,000</td><td></td></tr>"
                "<tr><td>연회비</td><td>2</td><td>2,000</td><td>2,000</td><td>1년 10,000원</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td>254,400</td><td></td></tr>"
                "</table>"
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>세목코드</td><td>돈</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>9911001</td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    messages = [item.message for item in response.tableChecks]
    assert any("소요예산 표 필수 항목 확인이 필요합니다" in message for message in messages)
    assert any("소요예산" in message for message in messages)
    assert all("표 구조가 HTML 표로 인식되지 않았습니다" not in message for message in messages)


def test_document_review_agent_does_not_flag_non_billing_table() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000007",
        traceId="00000000-0000-0000-0000-000000000008",
        conversationUid="00000000-0000-0000-0000-000000000009",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText="평가 결과를 보고합니다.  끝.",
            bodyHtml=(
                "<table>"
                "<tr><td>구분</td><td>금액</td><td>비고</td></tr>"
                "<tr><td>평가 항목</td><td>5점</td><td>우수</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert response.tableChecks == []


def test_document_review_agent_marks_table_review_available_when_no_tables() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000010",
        traceId="00000000-0000-0000-0000-000000000011",
        conversationUid="00000000-0000-0000-0000-000000000012",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(bodyText="정산 결과를 보고합니다.  끝."),
    )

    response = run_document_review_agent(request)

    assert response.tableChecksAvailable is True
    assert response.tableChecks == []


def test_document_review_agent_uses_declared_settlement_amount_not_other_amounts() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000013",
        traceId="00000000-0000-0000-0000-000000000014",
        conversationUid="00000000-0000-0000-0000-000000000015",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText=(
                "학술정보팀 수입 정산\n"
                "가. 정산 대상: 도서 연체료 192,400원, 문서 출력료 60,000원, 연회비 2,000원\n"
                "다. 정산 금액: 금254,400원\n"
                "라. 상세 내역\n"
                "마. 소요예산\n"
                "  끝."
            ),
            bodyHtml=(
                "<table>"
                "<tr><td>구분</td><td>건수</td><td>금액(원)</td><td></td><td>비고</td></tr>"
                "<tr><td>도서 연체료</td><td>180</td><td>192,400</td><td>192,400</td><td>1권 1일당 100원</td></tr>"
                "<tr><td>문서 출력료</td><td>5</td><td>60,000</td><td>60,000</td><td></td></tr>"
                "<tr><td>연회비</td><td>2</td><td>2,000</td><td>2,000</td><td>1년 10,000원</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td>254,400</td><td></td></tr>"
                "</table>"
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>세목코드</td><td>소요예산</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>9911001</td><td>254,400</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td></td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert response.tableChecks == []


def test_document_review_agent_ignores_one_won_table_difference() -> None:
    request = DocumentReviewRequest(
        queryUid="00000000-0000-0000-0000-000000000016",
        traceId="00000000-0000-0000-0000-000000000017",
        conversationUid="00000000-0000-0000-0000-000000000018",
        message="전자결재 문서를 검토해줘",
        document=ReviewDocument(
            bodyText="학술정보팀 수입 정산\n다. 정산 금액: 금254,401원\n라. 상세 내역\n마. 소요예산\n  끝.",
            bodyHtml=(
                "<table>"
                "<tr><td>구분</td><td>건수</td><td>금액(원)</td><td></td><td>비고</td></tr>"
                "<tr><td>도서 연체료</td><td>180</td><td>254,400</td><td>254,400</td><td></td></tr>"
                "<tr><td>합계</td><td></td><td></td><td>254,400</td><td></td></tr>"
                "</table>"
                "<table>"
                "<tr><td>회계연도</td><td>예산구분</td><td>세목</td><td>세목코드</td><td>소요예산</td></tr>"
                "<tr><td>2026학년도</td><td>학교회계</td><td>잡수입</td><td>9911001</td><td>254,400</td></tr>"
                "<tr><td>합계</td><td></td><td></td><td></td><td>254,400</td></tr>"
                "</table>"
            ),
        ),
    )

    response = run_document_review_agent(request)

    assert response.tableChecks == []


def test_table_check_response_contract_uses_camel_case_keys() -> None:
    item = TableCheckResponse(
        id="table-check-001",
        tableIndex=1,
        tableTitle="수입 정산 상세 내역",
        category="표 검토",
        severity="HIGH",
        status="CHECK_REQUIRED",
        message="금액 확인 필요",
        suggestion="원본 표에서 확인",
        evidence={"difference": 14920},
    )

    assert set(item.model_dump().keys()) == {
        "id",
        "tableIndex",
        "tableTitle",
        "category",
        "severity",
        "status",
        "message",
        "suggestion",
        "evidence",
    }
