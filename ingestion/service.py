from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from agents.library.db import create_library_engine
from agents.main_agent.embedding import get_embedding_provider
from ingestion.api.schemas import (
    CrawlStats,
    IngestionJobAcceptedResponse,
    IngestionJobStatusResponse,
    IngestionRunRequest,
    IngestionRunResponse,
    SourceIngestionResult,
    VectorStats,
)
from ingestion.chunking.markdown import load_markdown_chunks
from ingestion.crawlers.hansung_notice.config import (
    DEFAULT_OUTPUT_DIR as HANSUNG_OUTPUT_DIR,
    CrawlerConfig,
)
from ingestion.crawlers.hansung_notice.service import HansungNoticeCrawler
from ingestion.crawlers.hsel_library.config import (
    DEFAULT_OUTPUT_DIR as HSEL_OUTPUT_DIR,
    HselCrawlerConfig,
)
from ingestion.crawlers.hsel_library.service import HselLibraryCrawler
from ingestion.embedding.pgvector import init_main_agent_schema, upsert_chunks

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT_DIR / "data" / "schemas" / "main_agent.sql"
HANSUNG_DEFAULT_MAX_PAGES = 1
HSEL_DEFAULT_MAX_NOTICE_PAGES = 30

_job_store: dict[str, IngestionJobStatusResponse] = {}
_job_store_lock = Lock()
_run_lock = Lock()


def start_ingestion_job(request: IngestionRunRequest) -> IngestionJobAcceptedResponse:
    """Create an ingestion job record and return immediately for background execution."""
    job_id = f"ingestion-{datetime.now().astimezone().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    _set_job_status(IngestionJobStatusResponse(jobId=job_id, status="QUEUED"))
    return IngestionJobAcceptedResponse(
        jobId=job_id,
        status="ACCEPTED",
        statusUrl=f"/ingestion/status/{job_id}",
    )


def execute_ingestion_job(job_id: str, request: IngestionRunRequest) -> None:
    """Run a queued ingestion job in a background task."""
    if not _run_lock.acquire(blocking=False):
        _set_job_status(
            IngestionJobStatusResponse(
                jobId=job_id,
                status="FAILED",
                endedAt=datetime.now().astimezone().isoformat(),
                errorMessage="Another ingestion job is already running.",
            )
        )
        return

    started_at = datetime.now().astimezone()
    _set_job_status(
        IngestionJobStatusResponse(
            jobId=job_id,
            status="RUNNING",
            startedAt=started_at.isoformat(),
        )
    )
    try:
        result = run_ingestion(request, job_id=job_id, started_at=started_at)
        _set_job_status(
            IngestionJobStatusResponse(
                jobId=job_id,
                status=result.status,
                startedAt=result.startedAt,
                endedAt=result.endedAt,
                results=result.results,
            )
        )
    except Exception as error:
        _set_job_status(
            IngestionJobStatusResponse(
                jobId=job_id,
                status="FAILED",
                startedAt=started_at.isoformat(),
                endedAt=datetime.now().astimezone().isoformat(),
                errorMessage=str(error),
            )
        )
    finally:
        _run_lock.release()


def get_ingestion_job_status(job_id: str) -> IngestionJobStatusResponse | None:
    with _job_store_lock:
        return _job_store.get(job_id)


def run_ingestion(
    request: IngestionRunRequest,
    *,
    job_id: str | None = None,
    started_at: datetime | None = None,
) -> IngestionRunResponse:
    """Spring Scheduler가 호출하는 ingestion orchestration entrypoint."""
    started_at = started_at or datetime.now().astimezone()
    results: list[SourceIngestionResult] = []
    engine = create_library_engine()

    schema_initialized = False
    for source in request.sources:
        try:
            if source == "hansung_notice":
                result = _run_hansung_notice(
                    request,
                    engine=engine,
                    init_schema=request.initSchema and not schema_initialized,
                )
            else:
                result = _run_hsel_library(
                    request,
                    engine=engine,
                    init_schema=request.initSchema and not schema_initialized,
                )
            schema_initialized = schema_initialized or request.initSchema
            results.append(result)
        except Exception as error:
            results.append(
                SourceIngestionResult(
                    source=source,
                    status="FAILED",
                    crawl=CrawlStats(),
                    vector=VectorStats(),
                    errorMessage=str(error),
                )
            )

    ended_at = datetime.now().astimezone()
    return IngestionRunResponse(
        jobId=job_id or f"ingestion-{started_at.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
        status=_overall_status(results),
        startedAt=started_at.isoformat(),
        endedAt=ended_at.isoformat(),
        results=results,
    )


def _run_hansung_notice(
    request: IngestionRunRequest,
    *,
    engine,
    init_schema: bool,
) -> SourceIngestionResult:
    output_dir = HANSUNG_OUTPUT_DIR
    max_pages = request.maxPages or HANSUNG_DEFAULT_MAX_PAGES
    crawler = HansungNoticeCrawler(
        CrawlerConfig(
            output_dir=output_dir,
            since_date=request.sinceDate,
            max_pages=max_pages,
        )
    )
    crawl_result = crawler.run(
        category_keys=request.hansungCategories,
    )
    vector = _load_vectors(output_dir, source_prefix="notice", engine=engine, init_schema=init_schema)
    return SourceIngestionResult(
        source="hansung_notice",
        status="COMPLETED",
        crawl=_to_crawl_stats(crawl_result.to_dict()),
        vector=vector,
    )


def _run_hsel_library(
    request: IngestionRunRequest,
    *,
    engine,
    init_schema: bool,
) -> SourceIngestionResult:
    output_dir = HSEL_OUTPUT_DIR
    crawler = HselLibraryCrawler(
        HselCrawlerConfig(
            output_dir=output_dir,
            since_date=request.sinceDate,
            max_notice_pages=request.maxNoticePages or HSEL_DEFAULT_MAX_NOTICE_PAGES,
        )
    )
    crawl_result = crawler.run(crawl_pages=request.crawlPages, crawl_notices=request.crawlNotices)
    vector = _load_vectors(output_dir, source_prefix="library", engine=engine, init_schema=init_schema)
    return SourceIngestionResult(
        source="hsel_library",
        status="COMPLETED",
        crawl=_to_crawl_stats(crawl_result.to_dict()),
        vector=vector,
    )


def _load_vectors(output_dir: Path, *, source_prefix: str, engine, init_schema: bool) -> VectorStats:
    chunks = load_markdown_chunks(
        markdown_root=output_dir / "markdown",
        metadata_root=output_dir / "json",
        source_prefix=source_prefix,
    )
    if init_schema:
        init_main_agent_schema(engine, SCHEMA_PATH)

    embedding_provider = get_embedding_provider()
    result = upsert_chunks(
        engine=engine,
        chunks=chunks,
        embedding_provider=embedding_provider,
    )
    return VectorStats(
        insertedCount=result.inserted_count,
        updatedCount=result.updated_count,
        skippedCount=result.skipped_count,
        deletedCount=result.deleted_count,
        processedCount=result.processed_count,
        embeddingProvider=embedding_provider.__class__.__name__,
        embeddingDimensions=embedding_provider.dimensions,
    )


def _to_crawl_stats(payload: dict[str, Any]) -> CrawlStats:
    saved_count = (
        payload["saved_count"]
        if "saved_count" in payload
        else (payload.get("page_count") or 0) + (payload.get("notice_count") or 0)
    )
    return CrawlStats(
        savedCount=int(saved_count),
        updatedCount=int(payload.get("updated_count") or 0),
        unchangedCount=int(payload.get("unchanged_count") or 0),
        skipCount=int(payload.get("skip_count") or 0),
        errorCount=int(payload.get("error_count") or 0),
        processedPages=int(payload.get("processed_pages") or 0),
    )


def _overall_status(
    results: list[SourceIngestionResult],
) -> Literal["COMPLETED", "PARTIAL_FAILED", "FAILED"]:
    if all(result.status == "COMPLETED" for result in results):
        return "COMPLETED"
    if any(result.status == "COMPLETED" for result in results):
        return "PARTIAL_FAILED"
    return "FAILED"


def _set_job_status(status: IngestionJobStatusResponse) -> None:
    with _job_store_lock:
        _job_store[status.jobId] = status
