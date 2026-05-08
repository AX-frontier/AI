from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .models import Notice

SaveStatus = str


class Storage:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.markdown_dir = output_dir / "markdown"
        self.json_dir = output_dir / "json"
        self.errors_path = output_dir / "errors.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_saved(self, notice_id: str) -> bool:
        return any(self.markdown_dir.glob(f"*/{notice_id}.md"))

    def save_notice(self, notice: Notice) -> SaveStatus:
        record = notice.to_json_record()
        record.update(_build_hashes(notice))
        category_key = _safe_path_name(notice.source_category_key)
        markdown_dir = self.markdown_dir / category_key
        json_dir = self.json_dir / category_key
        markdown_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        status = self._detect_save_status(notice.notice_id, record["content_hash"])
        if status == "unchanged":
            return status

        (markdown_dir / f"{notice.notice_id}.md").write_text(
            self._render_markdown(notice),
            encoding="utf-8",
        )
        (json_dir / f"{notice.notice_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._save_kb_metadata(notice, json_dir)
        self._remove_stale_files(notice.notice_id, markdown_dir, json_dir)
        return status

    def log_error(self, stage: str, payload: dict[str, Any], error: Exception) -> None:
        self._append_jsonl(
            self.errors_path,
            {
                "stage": stage,
                "payload": payload,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n")

    def _render_markdown(self, notice: Notice) -> str:
        attachments = "\n".join(
            f"- [{attachment.name}]({attachment.url})" for attachment in notice.attachments
        )
        if not attachments:
            attachments = "- 없음"

        return (
            f"# {notice.title}\n\n"
            f"- 공지 ID: {notice.notice_id}\n"
            f"- 카테고리: {notice.category}\n"
            f"- 수집 목록: {notice.source_category_name}\n"
            f"- 작성 부서: {notice.department or ''}\n"
            f"- 작성일: {notice.published_at or ''}\n"
            f"- 조회수: {notice.views if notice.views is not None else ''}\n"
            f"- 원문 URL: {notice.source_url}\n"
            f"- 수집 시각: {notice.collected_at}\n\n"
            "## 첨부파일\n\n"
            f"{attachments}\n\n"
            "## 본문\n\n"
            f"{notice.content_markdown}\n"
        )

    def _save_kb_metadata(self, notice: Notice, json_dir: Path) -> None:
        attrs: dict[str, int | str] = {"doc_type": "notice"}
        notice_id = _to_int(notice.notice_id)
        if notice_id is not None:
            attrs["notice_id"] = notice_id
        if notice.title:
            attrs["title"] = notice.title
        if notice.category:
            attrs["category"] = notice.category
        if notice.department:
            attrs["department"] = notice.department
        if notice.published_at:
            attrs["posted_date"] = notice.published_at

        context = f"{notice.title}\n{notice.content_markdown}"
        academic_year = _extract_academic_year(context)
        if academic_year is not None:
            attrs["academic_year"] = academic_year
        semester = _extract_semester(context)
        if semester:
            attrs["semester"] = semester
        if notice.source_url:
            attrs["url"] = notice.source_url

        metadata_path = json_dir / f"{notice.notice_id}.md.metadata.json"
        metadata_path.write_text(
            json.dumps({"metadataAttributes": attrs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _detect_save_status(self, notice_id: str, content_hash: str) -> SaveStatus:
        existing_paths = list(self.json_dir.glob(f"*/{notice_id}.json"))
        if not existing_paths:
            return "created"
        for path in existing_paths:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if existing.get("content_hash") == content_hash:
                return "unchanged"
        return "updated"

    def _remove_stale_files(self, notice_id: str, markdown_dir: Path, json_dir: Path) -> None:
        current_paths = {
            markdown_dir / f"{notice_id}.md",
            json_dir / f"{notice_id}.json",
            json_dir / f"{notice_id}.md.metadata.json",
        }
        stale_paths = [
            *self.markdown_dir.glob(f"*/{notice_id}.md"),
            *self.json_dir.glob(f"*/{notice_id}.json"),
            *self.json_dir.glob(f"*/{notice_id}.md.metadata.json"),
        ]
        for path in stale_paths:
            if path not in current_paths:
                path.unlink(missing_ok=True)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _build_hashes(notice: Notice) -> dict[str, str]:
    body_hash = _sha256(_normalize_text(notice.content_markdown))
    attachments_hash = _sha256(
        "\n".join(f"{item.name}|{item.url}" for item in notice.attachments)
    )
    content_hash = _sha256(f"{notice.title}|{body_hash}|{attachments_hash}")
    return {
        "body_hash": body_hash,
        "attachments_hash": attachments_hash,
        "content_hash": content_hash,
    }


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_academic_year(text: str) -> int | None:
    match = re.search(r"(20\d{2})\s*학년도", text)
    return int(match.group(1)) if match else None


def _extract_semester(text: str) -> str | None:
    explicit = re.search(r"20\d{2}\s*학년도\s*([12])\s*학기", text)
    if explicit:
        return f"{explicit.group(1)}학기"
    operational = re.search(
        r"([12])\s*학기\s*(수강|개강|등록|휴학|복학|신청|성적|수업|학사|운영|일정)",
        text,
    )
    if operational:
        return f"{operational.group(1)}학기"
    if "여름학기" in text or "하계" in text:
        return "여름학기"
    if "겨울학기" in text or "동계" in text:
        return "겨울학기"
    return None


def _safe_path_name(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", value.strip())
    return safe.strip("_") or "uncategorized"
