from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from agents.main_agent.embedding import EmbeddingProvider
from ingestion.chunking.markdown import ChunkDocument


@dataclass(frozen=True, slots=True)
class VectorUpsertResult:
    inserted_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    deleted_count: int = 0

    @property
    def processed_count(self) -> int:
        return self.inserted_count + self.updated_count + self.skipped_count

    def to_dict(self) -> dict[str, int]:
        payload = asdict(self)
        payload["processed_count"] = self.processed_count
        return payload


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
) -> VectorUpsertResult:
    """Embed and upsert changed documents into main_agent.document_chunks."""
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

    result = VectorUpsertResult()
    with engine.begin() as connection:
        for document_id, document_chunks in _group_chunks_by_document(chunks).items():
            existing_hash = _load_existing_content_hash(connection, document_id)
            incoming_hash = _document_content_hash(document_chunks)
            if existing_hash and incoming_hash and existing_hash == incoming_hash:
                result = _add_result(result, skipped_count=len(document_chunks))
                continue

            deleted_count = 0
            if existing_hash is not None:
                delete_result = connection.execute(
                    text(
                        """
                        DELETE FROM main_agent.document_chunks
                        WHERE document_id = :document_id
                        """
                    ),
                    {"document_id": document_id},
                )
                deleted_count = delete_result.rowcount or 0

            for chunk in document_chunks:
                embedding = embedding_provider.embed_document(chunk.text)
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

            if existing_hash is None:
                result = _add_result(result, inserted_count=len(document_chunks))
            else:
                result = _add_result(
                    result,
                    updated_count=len(document_chunks),
                    deleted_count=deleted_count,
                )
    return result


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _group_chunks_by_document(chunks: list[ChunkDocument]) -> dict[str, list[ChunkDocument]]:
    grouped: dict[str, list[ChunkDocument]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, []).append(chunk)
    return grouped


def _load_existing_content_hash(connection, document_id: str) -> str | None:
    row = connection.execute(
        text(
            """
            SELECT metadata->>'content_hash' AS content_hash
            FROM main_agent.document_chunks
            WHERE document_id = :document_id
            ORDER BY chunk_id
            LIMIT 1
            """
        ),
        {"document_id": document_id},
    ).mappings().first()
    if row is None:
        return None
    return str(row["content_hash"]) if row["content_hash"] else ""


def _document_content_hash(chunks: list[ChunkDocument]) -> str | None:
    if not chunks:
        return None
    value = chunks[0].metadata.get("content_hash")
    return str(value) if value else None


def _add_result(
    result: VectorUpsertResult,
    *,
    inserted_count: int = 0,
    updated_count: int = 0,
    skipped_count: int = 0,
    deleted_count: int = 0,
) -> VectorUpsertResult:
    return VectorUpsertResult(
        inserted_count=result.inserted_count + inserted_count,
        updated_count=result.updated_count + updated_count,
        skipped_count=result.skipped_count + skipped_count,
        deleted_count=result.deleted_count + deleted_count,
    )
