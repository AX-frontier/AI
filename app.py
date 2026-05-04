from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from agents.library.api.router import router as library_router

load_dotenv()

app = FastAPI(title="AX-Prontier AI", version="0.1.0")
app.include_router(library_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    host = os.getenv("LIBRARY_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("LIBRARY_AGENT_PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)

