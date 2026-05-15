from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.library.db import create_library_engine
from agents.main_agent.embedding import get_embedding_provider
from ingestion.chunking.markdown import ChunkDocument, split_markdown_text
from ingestion.embedding.pgvector import init_schema, upsert_library_guide_chunks

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "raw" / "hsel-library" / "manual"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "data" / "schemas" / "library.sql"


def main() -> None:
    args = _parse_args()
    chunks = _load_manual_chunks(args.input_dir, max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    if args.dry_run:
        print(json.dumps({"input_dir": str(args.input_dir), "chunk_count": len(chunks), "sample": chunks[0].text if chunks else None}, ensure_ascii=False, indent=2))
        return

    engine = create_library_engine()
    if args.init_schema:
        init_schema(engine, args.schema_path)

    embedding_provider = get_embedding_provider()
    result = upsert_library_guide_chunks(engine=engine, chunks=chunks, embedding_provider=embedding_provider)
    print(json.dumps({"input_dir": str(args.input_dir), "embedding_provider": embedding_provider.__class__.__name__, "embedding_dimensions": embedding_provider.dimensions, **result.to_dict()}, ensure_ascii=False, indent=2))


def _load_manual_chunks(input_dir: Path, *, max_chars: int, overlap_chars: int) -> list[ChunkDocument]:
    chunks: list[ChunkDocument] = []
    for markdown_path in sorted(input_dir.glob("*.md")):
        title = _title_from_markdown(markdown_path)
        document_id = f"manual-{markdown_path.stem}"
        text = markdown_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for chunk_index, chunk_text in enumerate(split_markdown_text(text, title=title, max_chars=max_chars, overlap_chars=overlap_chars)):
            chunks.append(
                ChunkDocument(
                    chunk_id=f"library-manual-{markdown_path.stem}-{chunk_index:04d}",
                    document_id=document_id,
                    text=chunk_text,
                    metadata={
                        "document_id": document_id,
                        "title": title,
                        "url": f"urn:hsel-library:manual:{markdown_path.stem}",
                        "source_type": "manual_supplement",
                        "chunk_index": chunk_index,
                    },
                )
            )
    return chunks


def _title_from_markdown(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return path.stem.replace("-", " ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsert manually transcribed HSEL library guide documents.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--init-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap-chars", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    main()
