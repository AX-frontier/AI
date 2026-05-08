from __future__ import annotations

from fastapi.testclient import TestClient

from app import app
from ingestion.api.router import get_ingestion_executor, get_ingestion_starter
from ingestion.api.schemas import (
    CrawlStats,
    IngestionJobAcceptedResponse,
    IngestionRunRequest,
    SourceIngestionResult,
    VectorStats,
)
from ingestion.service import _set_job_status
from ingestion.api.schemas import IngestionJobStatusResponse


def test_ingestion_run_api_accepts_background_job() -> None:
    captured_request: IngestionRunRequest | None = None
    captured_job_id: str | None = None

    def fake_starter(request: IngestionRunRequest) -> IngestionJobAcceptedResponse:
        nonlocal captured_request
        captured_request = request
        return IngestionJobAcceptedResponse(
            jobId="ingestion-test",
            status="ACCEPTED",
            statusUrl="/ingestion/status/ingestion-test",
        )

    def fake_executor(job_id: str, request: IngestionRunRequest) -> None:
        nonlocal captured_job_id
        captured_job_id = job_id

    app.dependency_overrides[get_ingestion_starter] = lambda: fake_starter
    app.dependency_overrides[get_ingestion_executor] = lambda: fake_executor
    try:
        client = TestClient(app)
        response = client.post(
            "/ingestion/run",
            json={
                "sources": ["hansung_notice"],
                "sinceDate": "2025-11-08",
                "maxPages": 2,
                "initSchema": False,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202
    payload = response.json()
    assert payload["jobId"] == "ingestion-test"
    assert payload["status"] == "ACCEPTED"
    assert payload["statusUrl"] == "/ingestion/status/ingestion-test"
    assert captured_request is not None
    assert captured_job_id == "ingestion-test"
    assert captured_request.sources == ["hansung_notice"]
    assert captured_request.sinceDate.isoformat() == "2025-11-08"
    assert captured_request.maxPages == 2
    assert captured_request.initSchema is False


def test_ingestion_status_api_returns_completed_result() -> None:
    _set_job_status(
        IngestionJobStatusResponse(
            jobId="ingestion-completed",
            status="COMPLETED",
            startedAt="2026-05-08T00:00:00+09:00",
            endedAt="2026-05-08T00:00:01+09:00",
            results=[
                SourceIngestionResult(
                    source="hsel_library",
                    status="COMPLETED",
                    crawl=CrawlStats(savedCount=1, updatedCount=2, unchangedCount=3),
                    vector=VectorStats(
                        insertedCount=4,
                        updatedCount=5,
                        skippedCount=6,
                        deletedCount=7,
                        processedCount=15,
                        embeddingProvider="E5EmbeddingProvider",
                        embeddingDimensions=384,
                    ),
                )
            ],
        )
    )

    client = TestClient(app)
    response = client.get("/ingestion/status/ingestion-completed")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "COMPLETED"
    assert payload["results"][0]["source"] == "hsel_library"
    assert payload["results"][0]["vector"]["embeddingDimensions"] == 384


def test_ingestion_status_api_returns_404_for_missing_job() -> None:
    client = TestClient(app)
    response = client.get("/ingestion/status/missing-job")

    assert response.status_code == 404


def test_ingestion_run_api_rejects_empty_sources() -> None:
    client = TestClient(app)
    response = client.post("/ingestion/run", json={"sources": []})

    assert response.status_code == 422
