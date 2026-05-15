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


@dataclass(frozen=True, slots=True)
class LibraryGuideUpsertResult:
    inserted_doc_count: int = 0
    updated_doc_count: int = 0
    skipped_doc_count: int = 0
    inserted_chunk_count: int = 0
    updated_chunk_count: int = 0
    skipped_chunk_count: int = 0
    deleted_chunk_count: int = 0

    @property
    def processed_doc_count(self) -> int:
        return self.inserted_doc_count + self.updated_doc_count + self.skipped_doc_count

    @property
    def processed_chunk_count(self) -> int:
        return self.inserted_chunk_count + self.updated_chunk_count + self.skipped_chunk_count

    def to_dict(self) -> dict[str, int]:
        payload = asdict(self)
        payload["processed_doc_count"] = self.processed_doc_count
        payload["processed_chunk_count"] = self.processed_chunk_count
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
) -> LibraryGuideUpsertResult:
    """Embed and upsert only changed HSEL guide docs into library.guide_docs/chunks."""
    chunks_by_document: dict[str, list[ChunkDocument]] = {}
    for chunk in chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

    result = LibraryGuideUpsertResult()
    with engine.begin() as connection:
        for document_chunks in chunks_by_document.values():
            document_chunks.sort(key=lambda chunk: int(chunk.metadata.get("chunk_index", 0)))
            document_chunks = _dedupe_document_chunks(document_chunks)
            first_chunk = document_chunks[0]
            metadata = first_chunk.metadata
            source_url = _guide_doc_source_url(metadata.get("url"), first_chunk.document_id)
            title = str(metadata.get("title") or first_chunk.document_id)
            content = "\n\n".join(chunk.text for chunk in document_chunks)
            content_hash = _sha256(content)
            existing_doc = _load_existing_guide_doc(
                connection=connection,
                source_url=source_url,
            )
            existing_chunk_count = 0
            if existing_doc:
                existing_chunk_count = _count_guide_doc_chunks(
                    connection=connection,
                    guide_doc_id=int(existing_doc["id"]),
                )
            if existing_doc and existing_chunk_count > 0 and _is_unchanged_guide_doc(existing_doc, content_hash):
                if _needs_guide_doc_metadata_update(existing_doc, title, content_hash):
                    _update_guide_doc_metadata(
                        connection=connection,
                        guide_doc_id=int(existing_doc["id"]),
                        title=title,
                        content=content,
                        content_hash=content_hash,
                    )
                result = _add_library_result(
                    result,
                    skipped_doc_count=1,
                    skipped_chunk_count=len(document_chunks),
                )
                continue

            guide_doc_id = _upsert_guide_doc(
                connection=connection,
                title=title,
                source_url=source_url,
                content=content,
                content_hash=content_hash,
            )

            delete_result = connection.execute(
                text("DELETE FROM library.guide_doc_chunks WHERE guide_doc_id = :guide_doc_id"),
                {"guide_doc_id": guide_doc_id},
            )
            deleted_chunk_count = delete_result.rowcount or 0
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
            if existing_doc is None:
                result = _add_library_result(
                    result,
                    inserted_doc_count=1,
                    inserted_chunk_count=len(document_chunks),
                    deleted_chunk_count=deleted_chunk_count,
                )
            else:
                result = _add_library_result(
                    result,
                    updated_doc_count=1,
                    updated_chunk_count=len(document_chunks),
                    deleted_chunk_count=deleted_chunk_count,
                )

    return result


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
    source_url: str,
    content: str,
    content_hash: str,
) -> int:
    return int(
        connection.execute(
            text(
                """
                INSERT INTO library.guide_docs (source_url, title, content, content_hash)
                VALUES (:source_url, :title, :content, :content_hash)
                ON CONFLICT (source_url) DO UPDATE SET
                  title = EXCLUDED.title,
                  content = EXCLUDED.content,
                  content_hash = EXCLUDED.content_hash,
                  updated_at = NOW()
                RETURNING id
                """
            ),
            {
                "source_url": str(source_url) if source_url else None,
                "title": title,
                "content": content,
                "content_hash": content_hash,
            },
        ).scalar_one()
    )


def _load_existing_guide_doc(
    *,
    connection: Connection,
    source_url: str,
) -> dict | None:
    row = connection.execute(
        text(
            """
            SELECT id, title, content, content_hash
            FROM library.guide_docs
            WHERE source_url = :source_url
            ORDER BY id ASC
            LIMIT 1
            """
        ),
        {"source_url": str(source_url)},
    ).mappings().first()
    return dict(row) if row is not None else None


def _guide_doc_source_url(source_url: object, document_id: str) -> str:
    if source_url:
        return str(source_url)
    return f"urn:hsel-library:{document_id}"


def _is_unchanged_guide_doc(existing_doc: dict, incoming_hash: str) -> bool:
    existing_hash = existing_doc.get("content_hash")
    if existing_hash:
        return str(existing_hash) == incoming_hash
    return _sha256(str(existing_doc.get("content") or "")) == incoming_hash


def _needs_guide_doc_metadata_update(existing_doc: dict, title: str, content_hash: str) -> bool:
    return str(existing_doc.get("title") or "") != title or str(existing_doc.get("content_hash") or "") != content_hash


def _update_guide_doc_metadata(
    *,
    connection: Connection,
    guide_doc_id: int,
    title: str,
    content: str,
    content_hash: str,
) -> None:
    connection.execute(
        text(
            """
            UPDATE library.guide_docs
            SET title = :title,
                content = :content,
                content_hash = :content_hash,
                updated_at = NOW()
            WHERE id = :id
            """
        ),
        {
            "id": guide_doc_id,
            "title": title,
            "content": content,
            "content_hash": content_hash,
        },
    )


def _count_guide_doc_chunks(
    *,
    connection: Connection,
    guide_doc_id: int,
) -> int:
    return int(
        connection.execute(
            text(
                """
                SELECT COUNT(*) AS chunk_count
                FROM library.guide_doc_chunks
                WHERE guide_doc_id = :guide_doc_id
                """
            ),
            {"guide_doc_id": guide_doc_id},
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


def _add_library_result(
    result: LibraryGuideUpsertResult,
    *,
    inserted_doc_count: int = 0,
    updated_doc_count: int = 0,
    skipped_doc_count: int = 0,
    inserted_chunk_count: int = 0,
    updated_chunk_count: int = 0,
    skipped_chunk_count: int = 0,
    deleted_chunk_count: int = 0,
) -> LibraryGuideUpsertResult:
    return LibraryGuideUpsertResult(
        inserted_doc_count=result.inserted_doc_count + inserted_doc_count,
        updated_doc_count=result.updated_doc_count + updated_doc_count,
        skipped_doc_count=result.skipped_doc_count + skipped_doc_count,
        inserted_chunk_count=result.inserted_chunk_count + inserted_chunk_count,
        updated_chunk_count=result.updated_chunk_count + updated_chunk_count,
        skipped_chunk_count=result.skipped_chunk_count + skipped_chunk_count,
        deleted_chunk_count=result.deleted_chunk_count + deleted_chunk_count,
    )
