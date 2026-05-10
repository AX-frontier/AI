from __future__ import annotations

from fastapi import APIRouter

from agents.document_review.agent import run_document_review_agent
from agents.document_review.api.schemas import DocumentReviewRequest, DocumentReviewResponse

router = APIRouter(prefix="/document-review", tags=["document-review"])


@router.post("/chat", response_model=DocumentReviewResponse)
def chat(request: DocumentReviewRequest) -> DocumentReviewResponse:
    """Spring에서 호출하는 Document Review Agent 채팅 endpoint."""
    return run_document_review_agent(request)
