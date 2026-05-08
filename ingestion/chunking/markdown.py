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
    clean_for_embedding: bool = False,
) -> list[ChunkDocument]:
    """Load paired Markdown and metadata JSON files into vector-store chunks."""
    chunks: list[ChunkDocument] = []
    for markdown_path in sorted(markdown_root.glob("*/*.md")):
        category = markdown_path.parent.name
        metadata_path = metadata_root / category / f"{markdown_path.name}.metadata.json"
        record_path = metadata_root / category / f"{markdown_path.stem}.json"
        metadata = _load_record_metadata(record_path)
        metadata.update(_load_metadata(metadata_path))
        document_id = str(
            metadata.get("notice_id")
            or metadata.get("document_id")
            or markdown_path.stem
        )
        raw_text = markdown_path.read_text(encoding="utf-8")
        text = (
            _clean_markdown_for_embedding(raw_text)
            if clean_for_embedding
            else _normalize_markdown(raw_text)
        )
        if not text:
            continue

        title = str(metadata.get("title") or markdown_path.stem)
        chunk_texts = (
            split_markdown_text(
                text,
                title=title,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
            if clean_for_embedding
            else split_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
        )
        if clean_for_embedding and not chunk_texts:
            chunk_texts = [f"문서 제목: {title}"]
        for chunk_index, chunk_text in enumerate(chunk_texts):
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_index"] = chunk_index
            _drop_duplicate_identifier_metadata(chunk_metadata, document_id)
            chunks.append(
                ChunkDocument(
                    chunk_id=f"{source_prefix}-{document_id}-{chunk_index:04d}",
                    document_id=document_id,
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )
    return chunks


def split_markdown_text(
    text: str,
    *,
    title: str,
    max_chars: int = 1000,
    overlap_chars: int = 100,
) -> list[str]:
    """Split cleaned Markdown by headings, paragraphs, and table rows."""
    _validate_split_options(max_chars=max_chars, overlap_chars=overlap_chars)
    normalized_title = _clean_heading_text(title)
    chunks: list[str] = []
    section: str | None = None
    paragraph_lines: list[str] = []
    table_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph = _normalize_markdown("\n".join(paragraph_lines))
        _append_text_chunks(
            chunks,
            paragraph,
            title=normalized_title,
            section=section,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        paragraph_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return
        _append_table_chunks(
            chunks,
            table_lines,
            title=normalized_title,
            section=section,
            max_chars=max_chars,
        )
        table_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = _parse_heading(line)
        if heading is not None:
            flush_paragraph()
            flush_table()
            if heading != normalized_title:
                section = heading
            continue

        if _is_table_line(line):
            flush_paragraph()
            table_lines.append(line)
            continue

        if not line:
            flush_paragraph()
            flush_table()
            continue

        flush_table()
        paragraph_lines.append(line)

    flush_paragraph()
    flush_table()
    return _pack_small_chunks([chunk for chunk in chunks if chunk], max_chars=max_chars)


def split_text(text: str, *, max_chars: int = 1200, overlap_chars: int = 150) -> list[str]:
    """Split text on paragraph boundaries while keeping a small overlap."""
    _validate_split_options(max_chars=max_chars, overlap_chars=overlap_chars)
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
        if len(current) > max_chars:
            chunks.extend(
                _split_long_text(
                    current,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
            )
            current = ""

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def _load_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    attrs = payload.get("metadataAttributes", payload)
    if not isinstance(attrs, dict):
        raise ValueError(f"Metadata must be an object: {path}")
    return attrs


def _load_record_metadata(path: Path) -> dict[str, Any]:
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
        "source": record.get("source"),
        "doc_type": record.get("doc_type"),
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
    record_document_id = record.get("notice_id") or record.get("document_id") or path.stem
    if record.get("notice_id"):
        metadata["notice_id"] = _to_int(record.get("notice_id")) or record.get("notice_id")
    else:
        metadata["source_document_id"] = str(record_document_id)
    return {key: value for key, value in metadata.items() if value not in (None, "", [])}


def _normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_markdown_for_embedding(text: str) -> str:
    text = _normalize_markdown(text)
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _is_metadata_line(line):
            continue
        if line == "## 본문":
            cleaned_lines.append("")
            continue
        if _is_image_only_line(line):
            continue
        if _is_local_link_list_line(line):
            continue
        line = _strip_markdown_link_urls(line)
        line = _strip_bare_urls(line)
        if not line and raw_line.strip():
            continue
        cleaned_lines.append(line)
    return _normalize_markdown("\n".join(cleaned_lines))


def _validate_split_options(*, max_chars: int, overlap_chars: int) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be zero or positive.")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")


def _append_text_chunks(
    chunks: list[str],
    text: str,
    *,
    title: str,
    section: str | None,
    max_chars: int,
    overlap_chars: int,
) -> None:
    prefix = _context_prefix(title=title, section=section)
    available = max_chars - len(prefix) - 2
    if available <= 50:
        raise ValueError("max_chars is too small for the title context.")
    for part in _split_long_text_by_sentence(
        text,
        max_chars=available,
        overlap_chars=overlap_chars,
    ):
        chunk = f"{prefix}\n\n{part}".strip()
        chunks.append(chunk[:max_chars].strip() if len(chunk) > max_chars else chunk)


def _append_table_chunks(
    chunks: list[str],
    lines: list[str],
    *,
    title: str,
    section: str | None,
    max_chars: int,
) -> None:
    prefix = _context_prefix(title=title, section=section)
    header = _table_header(lines)
    rows = lines[len(header) :] if header else lines
    if not rows:
        rows = lines
        header = []

    current_rows: list[str] = []
    for row in rows:
        candidate_rows = current_rows + [row]
        candidate = _format_table_chunk(prefix, header, candidate_rows)
        if len(candidate) <= max_chars:
            current_rows = candidate_rows
            continue

        if current_rows:
            chunks.append(_format_table_chunk(prefix, header, current_rows))
            row_candidate = _format_table_chunk(prefix, header, [row])
            if len(row_candidate) <= max_chars:
                current_rows = [row]
            else:
                _append_text_chunks(
                    chunks,
                    _table_row_to_text(row),
                    title=title,
                    section=section,
                    max_chars=max_chars,
                    overlap_chars=0,
                )
                current_rows = []
            continue

        _append_text_chunks(
            chunks,
            _table_row_to_text(row),
            title=title,
            section=section,
            max_chars=max_chars,
            overlap_chars=0,
        )
        current_rows = []

    if current_rows:
        chunks.append(_format_table_chunk(prefix, header, current_rows))


def _format_table_chunk(prefix: str, header: list[str], rows: list[str]) -> str:
    return f"{prefix}\n\n" + "\n".join(header + rows)


def _table_header(lines: list[str]) -> list[str]:
    if len(lines) >= 2 and _is_table_separator(lines[1]):
        return lines[:2]
    return lines[:1] if lines else []


def _split_long_text_by_sentence(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？다요함임됨음)\]])\s+", text)
        if part.strip()
    ]
    if len(sentences) == 1:
        return _split_long_text(text, max_chars=max_chars, overlap_chars=overlap_chars)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(
                _split_long_text(
                    sentence,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
            )
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        chunks.append(current.strip())
        overlap = _tail_overlap(current, overlap_chars)
        current = f"{overlap} {sentence}".strip() if overlap else sentence
        if len(current) > max_chars:
            chunks.extend(
                _split_long_text(
                    current,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
            )
            current = ""

    if current:
        chunks.append(current.strip())
    return chunks


def _pack_small_chunks(
    chunks: list[str],
    *,
    max_chars: int,
    min_chars: int = 350,
) -> list[str]:
    packed: list[str] = []
    current = ""
    separator = "\n\n---\n\n"
    for chunk in chunks:
        if not current:
            current = chunk
            continue

        normalized_chunk = _strip_repeated_context_prefix(current, chunk)
        candidate = f"{current}{separator}{normalized_chunk}"
        if len(current) < min_chars and len(candidate) <= max_chars:
            current = candidate
            continue

        packed.append(current)
        current = chunk

    if current:
        packed.append(current)
    return packed


def _strip_repeated_context_prefix(previous: str, chunk: str) -> str:
    previous_prefix, _ = _split_context_prefix(previous)
    chunk_prefix, chunk_body = _split_context_prefix(chunk)
    if previous_prefix and previous_prefix == chunk_prefix and chunk_body:
        return chunk_body
    return chunk


def _split_context_prefix(chunk: str) -> tuple[str, str]:
    lines = chunk.splitlines()
    prefix_lines: list[str] = []
    index = 0
    for line in lines:
        if line.startswith("문서 제목: ") or line.startswith("섹션: "):
            prefix_lines.append(line)
            index += 1
            continue
        break

    if not prefix_lines:
        return "", chunk

    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(prefix_lines), "\n".join(lines[index:]).strip()


def _context_prefix(*, title: str, section: str | None) -> str:
    lines = [f"문서 제목: {title}"]
    if section and section != title:
        lines.append(f"섹션: {section}")
    return "\n".join(lines)


def _parse_heading(line: str) -> str | None:
    match = re.match(r"^#{1,6}\s+(.+)$", line)
    if not match:
        return None
    return _clean_heading_text(match.group(1))


def _clean_heading_text(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_metadata_line(line: str) -> bool:
    return bool(
        re.match(
            r"^-\s*(문서 ID|문서 유형|카테고리|작성자|작성일|조회수|원문 URL|수집 시각)\s*:",
            line,
        )
    )


def _is_image_only_line(line: str) -> bool:
    return bool(
        re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", line)
        or re.fullmatch(r"<img\b[^>]*>", line, flags=re.IGNORECASE)
    )


def _strip_markdown_link_urls(line: str) -> str:
    line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)


def _strip_bare_urls(line: str) -> str:
    line = re.sub(r"https?://\S+", "", line)
    return re.sub(r"\s+", " ", line).strip()


def _is_table_line(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and "|" in line[1:-1]


def _is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line))


def _is_local_link_list_line(line: str) -> bool:
    return bool(re.fullmatch(r"-\s*\[[^\]]+\]\((?:#|/)[^)]+\)", line))


def _table_row_to_text(row: str) -> str:
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    return " / ".join(cell for cell in cells if cell)


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


def _drop_duplicate_identifier_metadata(metadata: dict[str, Any], document_id: str) -> None:
    for key in ("document_id", "source_document_id", "notice_id"):
        value = metadata.get(key)
        if value is not None and str(value) == document_id:
            metadata.pop(key, None)
