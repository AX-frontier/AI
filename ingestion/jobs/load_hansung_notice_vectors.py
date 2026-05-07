from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.library.db import create_library_engine
from agents.main_agent.embedding import DeterministicEmbeddingProvider
from ingestion.chunking.markdown import load_markdown_chunks
from ingestion.embedding.pgvector import init_main_agent_schema, upsert_chunks


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "raw" / "hansung-notice"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "data" / "schemas" / "main_agent.sql"


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir
    markdown_root = input_dir / "markdown"
    metadata_root = input_dir / "json"

    chunks = load_markdown_chunks(
        markdown_root=markdown_root,
        metadata_root=metadata_root,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "chunk_count": len(chunks),
                    "sample": _sample_chunk(chunks),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    engine = create_library_engine()
    if args.init_schema:
        init_main_agent_schema(engine, args.schema_path)

    inserted_count = upsert_chunks(
        engine=engine,
        chunks=chunks,
        embedding_provider=DeterministicEmbeddingProvider(dimensions=args.embedding_dimensions),
    )
    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "inserted_count": inserted_count,
                "embedding_dimensions": args.embedding_dimensions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Hansung notice chunks into pgvector.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--init-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=150)
    parser.add_argument("--embedding-dimensions", type=int, default=1536)
    return parser.parse_args()


def _sample_chunk(chunks: list) -> dict | None:
    if not chunks:
        return None
    chunk = chunks[0]
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text[:300],
        "metadata": chunk.metadata,
    }


if __name__ == "__main__":
    main()

