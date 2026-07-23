"""FaceHarmony AI FastAPI entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="FaceHarmony AI",
    description=(
        "Educational facial geometry analysis API. "
        "Scores describe measurable proportions and symmetry — not beauty."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    """Root welcome payload."""
    return {
        "name": "FaceHarmony AI",
        "docs": "/docs",
        "analyze": "POST /analyze",
    }
