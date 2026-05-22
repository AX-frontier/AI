from __future__ import annotations

import json
import re

from agents.document_review.models import CheckRequiredItem, FormatNoticeItem, RuleFinding, TableCheckItem
from agents.document_review.prompt_context import load_manual_prompt_context
from agents.document_review.tables import ExtractedTable
from agents.main_agent.llm.base import LLMClient
from agents.main_agent.llm.factory import get_llm_client

MAX_LLM_CHECK_ITEMS = 8
MAX_DOCUMENT_CHARS_FOR_LLM = 12000


def review_with_manual_prompts(
    *,
    body_text: str,
    revised_text: str,
    findings: list[RuleFinding],
    checks: list[CheckRequiredItem],
    format_notices: list[FormatNoticeItem],
    table_checks: list[TableCheckItem],
    extracted_tables: list[ExtractedTable],
    llm_client: LLMClient | None = None,
) -> list[CheckRequiredItem]:
    """두 HWP 프롬프트 원문을 LLM에 제공해 규칙 엔진 누락 항목만 보강한다."""
    manual_context = load_manual_prompt_context()
    if not manual_context:
        return []
    client = llm_client or _safe_default_llm_client()
    if client is None:
        return []

    prompt = _build_manual_review_prompt(
        manual_context=manual_context,
        body_text=body_text,
        revised_text=revised_text,
        findings=findings,
        checks=checks,
        format_notices=format_notices,
        table_checks=table_checks,
        extracted_tables=extracted_tables,
    )
    try:
        raw_response = client.generate(prompt)
    except Exception:
        return []
    return _dedupe_llm_checks(_parse_llm_check_items(raw_response), checks)


def _safe_default_llm_client() -> LLMClient | None:
    try:
        return get_llm_client()
    except Exception:
        return None


def _build_manual_review_prompt(
    *,
    manual_context: str,
    body_text: str,
    revised_text: str,
    findings: list[RuleFinding],
    checks: list[CheckRequiredItem],
    format_notices: list[FormatNoticeItem],
    table_checks: list[TableCheckItem],
    extracted_tables: list[ExtractedTable],
) -> str:
    existing_findings = "\n".join(
        f"- {finding.rule_code}: {finding.original_text} -> {finding.suggested_text or '확인 필요'}"
        for finding in findings[:40]
    ) or "- 없음"
    existing_checks = "\n".join(
        f"- {item.category}: {item.message}"
        for item in checks[:40]
    ) or "- 없음"
    existing_table_checks = "\n".join(
        f"- 표 {item.table_index} {item.table_title}: {item.message}"
        for item in table_checks[:20]
    ) or "- 없음"
    table_summary = "\n".join(
        f"- 표 {table.index}: {table.rowCount}행 x {table.columnCount}열"
        for table in extracted_tables[:20]
    ) or "- 없음"
    return f"""
당신은 대학 전자결재 문서 최종 검토자입니다.
아래 [전자결재 프롬프트 원문]은 사용자가 제공한 HWP 매뉴얼/프롬프트에서 추출한 텍스트입니다.
이 원문에 명시된 규칙만 근거로 삼고, 원문에 없는 규칙은 만들지 마세요.

역할:
1. 이미 규칙 엔진이 검출한 자동 수정/확인 항목을 중복으로 다시 내지 마세요.
2. 규칙 엔진이 놓친 항목이 있으면 "직접 확인 필요" 항목으로만 보강하세요.
3. 날짜, 금액, 부서명, 기관명, 문서번호, 세목코드 같은 사실관계는 임의로 확정하지 마세요.
4. 표 안의 값은 직접 수정 지시가 아니라 확인 항목으로만 제시하세요.
5. 출력은 반드시 JSON 객체 하나만 반환하세요.

반환 형식:
{{
  "checkRequiredItems": [
    {{
      "category": "항목 번호 체계",
      "message": "같은 단계의 항목 번호가 가., 가.로 반복되어 나. 여부 확인이 필요합니다.",
      "originalText": "가. 검토결과: ...",
      "lineStart": 4
    }}
  ]
}}

누락 항목이 없으면 {{"checkRequiredItems":[]}} 만 반환하세요.
최대 {MAX_LLM_CHECK_ITEMS}개만 반환하세요.

[전자결재 프롬프트 원문]
{manual_context}

[규칙 엔진이 이미 자동 수정으로 잡은 항목]
{existing_findings}

[규칙 엔진이 이미 직접 확인으로 잡은 항목]
{existing_checks}

[규칙 엔진이 이미 표 검토로 잡은 항목]
{existing_table_checks}

[추출된 표 요약]
{table_summary}

[원문]
{body_text[:MAX_DOCUMENT_CHARS_FOR_LLM]}

[규칙 엔진 반영 후 문서]
{revised_text[:MAX_DOCUMENT_CHARS_FOR_LLM]}
""".strip()


def _parse_llm_check_items(raw_response: str) -> list[CheckRequiredItem]:
    payload = _extract_json_object(raw_response)
    if payload is None:
        return []
    items = payload.get("checkRequiredItems")
    if not isinstance(items, list):
        return []
    parsed: list[CheckRequiredItem] = []
    for item in items[:MAX_LLM_CHECK_ITEMS]:
        if not isinstance(item, dict):
            continue
        category = _clean_string(item.get("category"), max_length=40)
        message = _clean_string(item.get("message"), max_length=300)
        if not category or not message:
            continue
        original_text = _clean_string(item.get("originalText"), max_length=180)
        line_start = _clean_line_start(item.get("lineStart"))
        parsed.append(
            CheckRequiredItem(
                category=category,
                message=message,
                original_text=original_text,
                line_start=line_start,
            )
        )
    return parsed


def _extract_json_object(raw_response: str) -> dict | None:
    text = raw_response.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _clean_string(value, *, max_length: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return None
    return text[:max_length]


def _clean_line_start(value) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _dedupe_llm_checks(
    llm_checks: list[CheckRequiredItem],
    existing_checks: list[CheckRequiredItem],
) -> list[CheckRequiredItem]:
    seen = {
        (item.category, item.message, item.original_text, item.line_start)
        for item in existing_checks
    }
    deduped: list[CheckRequiredItem] = []
    for item in llm_checks:
        key = (item.category, item.message, item.original_text, item.line_start)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
