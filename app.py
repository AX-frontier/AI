from __future__ import annotations

import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.document_review.api.router import router as document_review_router
from agents.library.api.router import router as library_router
from agents.main_agent.api.router import router as main_agent_router
from agents.orchestrator.api.router import router as orchestrator_router
from ingestion.api.router import router as ingestion_router

load_dotenv()
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="AX-Prontier AI", version="0.1.0")
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
origin_list = [origin.strip() for origin in allowed_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin_list,
    allow_credentials="*" not in origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(library_router)
app.include_router(main_agent_router)
app.include_router(orchestrator_router)
app.include_router(ingestion_router)
app.include_router(document_review_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    host = os.getenv("LIBRARY_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("LIBRARY_AGENT_PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
