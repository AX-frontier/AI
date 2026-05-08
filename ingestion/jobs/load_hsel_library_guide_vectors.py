from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.library.db import create_library_engine
from agents.main_agent.embedding import get_embedding_provider
from ingestion.chunking.markdown import load_markdown_chunks
from ingestion.embedding.pgvector import init_schema, upsert_library_guide_chunks


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "raw" / "hsel-library"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "data" / "schemas" / "library.sql"


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir
    chunks = load_markdown_chunks(
        markdown_root=input_dir / "markdown",
        metadata_root=input_dir / "json",
        source_prefix="library",
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        clean_for_embedding=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "stats": _chunk_stats(chunks),
                    "sample": _sample_chunk(chunks),
                    "longest_samples": _longest_samples(chunks),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    engine = create_library_engine()
    if args.init_schema:
        init_schema(engine, args.schema_path)

    embedding_provider = get_embedding_provider()
    guide_doc_count, chunk_count = upsert_library_guide_chunks(
        engine=engine,
        chunks=chunks,
        embedding_provider=embedding_provider,
    )
    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "guide_doc_count": guide_doc_count,
                "chunk_count": chunk_count,
                "embedding_provider": embedding_provider.__class__.__name__,
                "embedding_dimensions": embedding_provider.dimensions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load HSEL library guide documents into library pgvector tables."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--init-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap-chars", type=int, default=100)
    return parser.parse_args()


def _sample_chunk(chunks: list) -> dict | None:
    if not chunks:
        return None
    chunk = chunks[0]
    return {
        "document_id": chunk.document_id,
        "text": chunk.text[:300],
        "metadata": chunk.metadata,
    }


def _chunk_stats(chunks: list) -> dict:
    lengths = [len(chunk.text) for chunk in chunks]
    if not lengths:
        return {
            "chunk_count": 0,
            "doc_count": 0,
            "min_len": 0,
            "max_len": 0,
            "avg_len": 0,
            "under_200": 0,
            "over_1000": 0,
            "metadata_header_chunks": 0,
            "image_link_chunks": 0,
            "table_chunks": 0,
        }
    return {
        "chunk_count": len(chunks),
        "doc_count": len({chunk.document_id for chunk in chunks}),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "under_200": sum(1 for length in lengths if length < 200),
        "over_1000": sum(1 for length in lengths if length > 1000),
        "metadata_header_chunks": sum(1 for chunk in chunks if "- 문서 ID:" in chunk.text),
        "image_link_chunks": sum(1 for chunk in chunks if "![](" in chunk.text),
        "table_chunks": sum(1 for chunk in chunks if "| ---" in chunk.text),
    }


def _longest_samples(chunks: list, *, limit: int = 3) -> list[dict]:
    return [
        {
            "document_id": chunk.document_id,
            "title": chunk.metadata.get("title"),
            "chunk_index": chunk.metadata.get("chunk_index"),
            "length": len(chunk.text),
            "text": chunk.text[:500],
        }
        for chunk in sorted(chunks, key=lambda item: len(item.text), reverse=True)[:limit]
    ]


if __name__ == "__main__":
    main()
