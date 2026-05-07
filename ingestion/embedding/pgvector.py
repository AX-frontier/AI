from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from agents.main_agent.embedding import EmbeddingProvider
from ingestion.chunking.markdown import ChunkDocument


def init_main_agent_schema(engine: Engine, schema_path: Path) -> None:
    """Create pgvector schema/table/indexes used by Main Agent retrieval."""
    sql = schema_path.read_text(encoding="utf-8")
    with engine.begin() as connection:
        connection.execute(text(sql))


def upsert_chunks(
    *,
    engine: Engine,
    chunks: list[ChunkDocument],
    embedding_provider: EmbeddingProvider,
) -> int:
    """Embed and upsert chunks into main_agent.document_chunks."""
    statement = text(
        """
        INSERT INTO main_agent.document_chunks (
          chunk_id,
          document_id,
          text,
          embedding,
          metadata,
          updated_at
        )
        VALUES (
          :chunk_id,
          :document_id,
          :text,
          CAST(:embedding AS vector),
          CAST(:metadata AS jsonb),
          NOW()
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
          document_id = EXCLUDED.document_id,
          text = EXCLUDED.text,
          embedding = EXCLUDED.embedding,
          metadata = EXCLUDED.metadata,
          updated_at = NOW()
        """
    )

    with engine.begin() as connection:
        for chunk in chunks:
            embedding = embedding_provider.embed_query(chunk.text)
            connection.execute(
                statement,
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "embedding": _vector_literal(embedding),
                    "metadata": json.dumps(chunk.metadata, ensure_ascii=False),
                },
            )
    return len(chunks)


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"

