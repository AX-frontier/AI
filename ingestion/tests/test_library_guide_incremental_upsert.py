from __future__ import annotations

from ingestion.embedding.pgvector import (
    LibraryGuideUpsertResult,
    _add_library_result,
    _guide_doc_source_url,
    _is_unchanged_guide_doc,
    _needs_guide_doc_metadata_update,
)


def test_library_guide_unchanged_detection_uses_persisted_content_hash() -> None:
    existing = {
        "id": 1,
        "content": "old content",
        "content_hash": "abc123",
    }

    assert _is_unchanged_guide_doc(existing, "abc123") is True
    assert _is_unchanged_guide_doc(existing, "changed") is False


def test_library_guide_unchanged_detection_backfills_missing_hash_from_content() -> None:
    existing = {
        "id": 1,
        "content": "학술정보관 개관시간 안내",
        "content_hash": None,
    }
    incoming_hash = "d4d8c91eeda740e81d368eb09cff0910b24990ba01c4dce9e69d282f1726988c"

    assert _is_unchanged_guide_doc(existing, incoming_hash) is True


def test_library_guide_source_url_falls_back_to_stable_document_urn() -> None:
    assert _guide_doc_source_url("", "library-guide-001") == "urn:hsel-library:library-guide-001"
    assert _guide_doc_source_url(None, "library-guide-001") == "urn:hsel-library:library-guide-001"
    assert _guide_doc_source_url("https://hsel.hansung.ac.kr/page", "library-guide-001") == (
        "https://hsel.hansung.ac.kr/page"
    )


def test_library_guide_metadata_update_needed_for_missing_hash_or_changed_title() -> None:
    existing = {"title": "개관시간", "content_hash": None}

    assert _needs_guide_doc_metadata_update(existing, "개관시간", "abc123") is True
    assert _needs_guide_doc_metadata_update({"title": "이전 제목", "content_hash": "abc123"}, "개관시간", "abc123") is True
    assert _needs_guide_doc_metadata_update({"title": "개관시간", "content_hash": "abc123"}, "개관시간", "abc123") is False


def test_library_guide_upsert_result_accumulates_document_and_chunk_counts() -> None:
    result = _add_library_result(
        LibraryGuideUpsertResult(inserted_doc_count=1, inserted_chunk_count=3),
        updated_doc_count=1,
        skipped_doc_count=2,
        updated_chunk_count=4,
        skipped_chunk_count=5,
        deleted_chunk_count=6,
    )

    assert result.inserted_doc_count == 1
    assert result.updated_doc_count == 1
    assert result.skipped_doc_count == 2
    assert result.processed_doc_count == 4
    assert result.processed_chunk_count == 12
    assert result.deleted_chunk_count == 6
