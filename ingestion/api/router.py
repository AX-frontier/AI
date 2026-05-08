from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from ingestion.api.schemas import (
    IngestionJobAcceptedResponse,
    IngestionJobStatusResponse,
    IngestionRunRequest,
)
from ingestion.service import execute_ingestion_job, get_ingestion_job_status, start_ingestion_job

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


def get_ingestion_starter() -> Callable[[IngestionRunRequest], IngestionJobAcceptedResponse]:
    return start_ingestion_job


def get_ingestion_executor() -> Callable[[str, IngestionRunRequest], None]:
    return execute_ingestion_job


@router.post(
    "/run",
    response_model=IngestionJobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def run(
    request: IngestionRunRequest,
    background_tasks: BackgroundTasks,
    starter: Callable[[IngestionRunRequest], IngestionJobAcceptedResponse] = Depends(get_ingestion_starter),
    executor: Callable[[str, IngestionRunRequest], None] = Depends(get_ingestion_executor),
) -> IngestionJobAcceptedResponse:
    """Spring Scheduler에서 호출하는 크롤링/pgvector 적재 실행 endpoint."""
    accepted = starter(request)
    background_tasks.add_task(executor, accepted.jobId, request)
    return accepted


@router.get("/status/{job_id}", response_model=IngestionJobStatusResponse)
def status_by_job_id(job_id: str) -> IngestionJobStatusResponse:
    """Spring이 background ingestion job 상태를 polling할 때 사용한다."""
    job_status = get_ingestion_job_status(job_id)
    if job_status is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return job_status
