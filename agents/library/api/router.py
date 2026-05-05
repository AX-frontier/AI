from __future__ import annotations

from fastapi import APIRouter, Depends

from agents.library.agent import run_library_agent
from agents.library.api.schemas import LibraryChatRequest, LibraryChatResponse
from agents.library.repository import LibraryRepository, get_library_repository

router = APIRouter(prefix="/library", tags=["library"])


@router.post("/chat", response_model=LibraryChatResponse)
def chat(
    request: LibraryChatRequest,
    repository: LibraryRepository = Depends(get_library_repository),
) -> LibraryChatResponse:
    """Spring에서 호출하는 Library Agent 채팅 endpoint."""
    return run_library_agent(request, repository=repository)
