from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .models import HselDocument

SaveStatus = str


class HselStorage:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.markdown_dir = output_dir / "markdown"
        self.json_dir = output_dir / "json"
        self.errors_path = output_dir / "errors.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_saved(self, doc_type: str, document_id: str) -> bool:
        return (self.markdown_dir / doc_type / f"{document_id}.md").exists()

    def save_document(self, document: HselDocument) -> SaveStatus:
        record = document.to_json_record()
        record.update(_build_hashes(document))
        markdown_dir = self.markdown_dir / document.doc_type
        json_dir = self.json_dir / document.doc_type
        markdown_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        status = self._detect_save_status(document.doc_type, document.document_id, record["content_hash"])
        if status == "unchanged":
            return status

        (markdown_dir / f"{document.document_id}.md").write_text(
            self._render_markdown(document),
            encoding="utf-8",
        )
        (json_dir / f"{document.document_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._save_metadata(document, record, json_dir)
        return status

    def log_error(self, stage: str, payload: dict[str, Any], error: Exception) -> None:
        self.errors_path.parent.mkdir(parents=True, exist_ok=True)
        with self.errors_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "stage": stage,
                        "payload": payload,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _render_markdown(self, document: HselDocument) -> str:
        return (
            f"# {document.title}\n\n"
            f"- 문서 ID: {document.document_id}\n"
            f"- 문서 유형: {document.doc_type}\n"
            f"- 카테고리: {document.category or ''}\n"
            f"- 작성자: {document.author or ''}\n"
            f"- 작성일: {document.published_at or ''}\n"
            f"- 조회수: {document.views if document.views is not None else ''}\n"
            f"- 원문 URL: {document.source_url}\n"
            f"- 수집 시각: {document.collected_at}\n\n"
            "## 본문\n\n"
            f"{document.content_markdown}\n"
        )

    def _save_metadata(
        self,
        document: HselDocument,
        record: dict[str, object],
        json_dir: Path,
    ) -> None:
        attrs = {
            "source": "hsel_library",
            "doc_type": document.doc_type,
            "title": document.title,
            "category": document.category,
            "posted_date": document.published_at,
            "author": document.author,
            "views": document.views,
            "url": document.source_url,
            "content_path": document.content_path,
            "body_hash": record.get("body_hash"),
            "content_hash": record.get("content_hash"),
        }
        attrs = {key: value for key, value in attrs.items() if value not in (None, "", [])}
        (json_dir / f"{document.document_id}.md.metadata.json").write_text(
            json.dumps({"metadataAttributes": attrs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _detect_save_status(
        self,
        doc_type: str,
        document_id: str,
        content_hash: str,
    ) -> SaveStatus:
        record_path = self.json_dir / doc_type / f"{document_id}.json"
        if not record_path.exists():
            return "created"
        try:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "updated"
        if existing.get("content_hash") == content_hash:
            return "unchanged"
        return "updated"


def _build_hashes(document: HselDocument) -> dict[str, str]:
    body_hash = _sha256(_normalize_text(document.content_markdown))
    content_hash = _sha256(f"{document.title}|{body_hash}")
    return {"body_hash": body_hash, "content_hash": content_hash}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
