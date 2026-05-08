from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

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


def init_schema(engine: Engine, schema_path: Path) -> None:
    """Run a schema SQL file."""
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


def upsert_library_guide_chunks(
    *,
    engine: Engine,
    chunks: list[ChunkDocument],
    embedding_provider: EmbeddingProvider,
) -> tuple[int, int]:
    """Embed and upsert HSEL guide docs into library.guide_docs/chunks."""
    chunks_by_document: dict[str, list[ChunkDocument]] = {}
    for chunk in chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

    guide_doc_count = 0
    chunk_count = 0
    with engine.begin() as connection:
        for document_chunks in chunks_by_document.values():
            document_chunks.sort(key=lambda chunk: int(chunk.metadata.get("chunk_index", 0)))
            document_chunks = _dedupe_document_chunks(document_chunks)
            first_chunk = document_chunks[0]
            metadata = first_chunk.metadata
            guide_doc_id = _upsert_guide_doc(
                connection=connection,
                title=str(metadata.get("title") or first_chunk.document_id),
                source_url=metadata.get("url"),
                content="\n\n".join(chunk.text for chunk in document_chunks),
            )
            guide_doc_count += 1

            connection.execute(
                text("DELETE FROM library.guide_doc_chunks WHERE guide_doc_id = :guide_doc_id"),
                {"guide_doc_id": guide_doc_id},
            )
            for chunk_index, chunk in enumerate(document_chunks):
                embedding = embedding_provider.embed_document(chunk.text)
                connection.execute(
                    _UPSERT_LIBRARY_GUIDE_CHUNK,
                    {
                        "guide_doc_id": guide_doc_id,
                        "chunk_index": chunk_index,
                        "content": chunk.text,
                        "embedding": _vector_literal(embedding),
                        "content_hash": _sha256(chunk.text),
                    },
                )
                chunk_count += 1

    return guide_doc_count, chunk_count


def _dedupe_document_chunks(chunks: list[ChunkDocument]) -> list[ChunkDocument]:
    deduped: list[ChunkDocument] = []
    seen_hashes: set[str] = set()
    for chunk in chunks:
        content_hash = _sha256(chunk.text)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        deduped.append(chunk)
    return deduped


_UPSERT_LIBRARY_GUIDE_CHUNK = text(
    """
    INSERT INTO library.guide_doc_chunks (
      guide_doc_id,
      chunk_index,
      content,
      embedding,
      content_hash,
      updated_at
    )
    VALUES (
      :guide_doc_id,
      :chunk_index,
      :content,
      CAST(:embedding AS vector),
      :content_hash,
      NOW()
    )
    ON CONFLICT (guide_doc_id, chunk_index) DO UPDATE SET
      content = EXCLUDED.content,
      embedding = EXCLUDED.embedding,
      content_hash = EXCLUDED.content_hash,
      updated_at = NOW()
    """
)


def _upsert_guide_doc(
    *,
    connection: Connection,
    title: str,
    source_url: object,
    content: str,
) -> int:
    existing_id = None
    if source_url:
        existing_id = connection.execute(
            text(
                """
                SELECT id
                FROM library.guide_docs
                WHERE source_url = :source_url
                ORDER BY id ASC
                LIMIT 1
                """
            ),
            {"source_url": str(source_url)},
        ).scalar_one_or_none()

    if existing_id is not None:
        return int(
            connection.execute(
                text(
                    """
                    UPDATE library.guide_docs
                    SET title = :title,
                        content = :content,
                        updated_at = NOW()
                    WHERE id = :id
                    RETURNING id
                    """
                ),
                {"id": existing_id, "title": title, "content": content},
            ).scalar_one()
        )

    return int(
        connection.execute(
            text(
                """
                INSERT INTO library.guide_docs (source_url, title, content)
                VALUES (:source_url, :title, :content)
                RETURNING id
                """
            ),
            {
                "source_url": str(source_url) if source_url else None,
                "title": title,
                "content": content,
            },
        ).scalar_one()
    )


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _group_chunks_by_document(chunks: list[ChunkDocument]) -> dict[str, list[ChunkDocument]]:
    grouped: dict[str, list[ChunkDocument]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, []).append(chunk)
    return grouped


def _load_existing_content_hash(connection: Connection, document_id: str) -> str | None:
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
