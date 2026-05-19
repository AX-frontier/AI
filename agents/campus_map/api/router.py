from __future__ import annotations

from fastapi import APIRouter, Depends

from agents.campus_map.agent import run_campus_map_agent
from agents.campus_map.api.schemas import CampusMapRequest, CampusMapResponse
from agents.campus_map.repository import CampusMapRepository, get_campus_map_repository

router = APIRouter(prefix="/campus-map", tags=["campus-map"])


def get_repository() -> CampusMapRepository:
    return get_campus_map_repository()


@router.post("/chat", response_model=CampusMapResponse)
def chat(
    request: CampusMapRequest,
    repository: CampusMapRepository = Depends(get_repository),
) -> CampusMapResponse:
    return run_campus_map_agent(request, repository=repository)
