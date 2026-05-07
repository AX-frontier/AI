from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ChunkDocument:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


def load_markdown_chunks(
    *,
    markdown_root: Path,
    metadata_root: Path,
    source_prefix: str = "notice",
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[ChunkDocument]:
    """Load paired Markdown and metadata JSON files into vector-store chunks."""
    chunks: list[ChunkDocument] = []
    for markdown_path in sorted(markdown_root.glob("*/*.md")):
        category = markdown_path.parent.name
        metadata_path = metadata_root / category / f"{markdown_path.name}.metadata.json"
        record_path = metadata_root / category / f"{markdown_path.stem}.json"
        metadata = _load_notice_metadata(record_path)
        metadata.update(_load_metadata(metadata_path))
        document_id = str(
            metadata.get("notice_id")
            or metadata.get("document_id")
            or markdown_path.stem
        )
        text = _normalize_markdown(markdown_path.read_text(encoding="utf-8"))
        if not text:
            continue

        for chunk_index, chunk_text in enumerate(
            split_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
        ):
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata.pop("document_id", None)
            chunks.append(
                ChunkDocument(
                    chunk_id=f"{source_prefix}-{document_id}-{chunk_index:04d}",
                    document_id=document_id,
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )
    return chunks


def split_text(text: str, *, max_chars: int = 1200, overlap_chars: int = 150) -> list[str]:
    """Split text on paragraph boundaries while keeping a small overlap."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be zero or positive.")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(
                _split_long_text(
                    paragraph,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
            )
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        chunks.append(current.strip())
        overlap = _tail_overlap(current, overlap_chars)
        current = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def _load_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    attrs = payload.get("metadataAttributes", payload)
    if not isinstance(attrs, dict):
        raise ValueError(f"Metadata must be an object: {path}")
    return attrs


def _load_notice_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    record = json.loads(path.read_text(encoding="utf-8"))
    attachments = record.get("attachments") or []
    attachment_names = [
        str(item.get("name"))
        for item in attachments
        if isinstance(item, dict) and item.get("name")
    ]
    attachment_urls = [
        str(item.get("url"))
        for item in attachments
        if isinstance(item, dict) and item.get("url")
    ]

    metadata: dict[str, Any] = {
        "doc_type": "notice",
        "source": record.get("source") or "hansung_notice",
        "notice_id": _to_int(record.get("notice_id")) or record.get("notice_id") or path.stem,
        "title": record.get("title"),
        "category": record.get("category"),
        "source_category_key": record.get("source_category_key"),
        "source_category_name": record.get("source_category_name"),
        "department": record.get("department"),
        "posted_date": record.get("published_at"),
        "views": record.get("views"),
        "url": record.get("source_url"),
        "content_path": record.get("content_path"),
        "collected_at": record.get("collected_at"),
        "attachment_count": len(attachments),
        "attachment_names": attachment_names,
        "attachment_urls": attachment_urls,
        "body_hash": record.get("body_hash"),
        "attachments_hash": record.get("attachments_hash"),
        "content_hash": record.get("content_hash"),
    }
    return {key: value for key, value in metadata.items() if value not in (None, "", [])}


def _normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def _tail_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars == 0:
        return ""
    tail = text[-overlap_chars:].strip()
    first_space = tail.find(" ")
    return tail[first_space + 1 :].strip() if first_space > 0 else tail


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
