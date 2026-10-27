"""FastAPI inference service.

Run locally:  uvicorn verasonus.api.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs

Privacy by default: an uploaded file is written to a temporary file and deleted as soon as
its analysis finishes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from verasonus import __version__
from verasonus.api.schemas import AnalysisResult, HealthResponse

app = FastAPI(
    title="VeraSonus API",
    version=__version__,
    description="Estimates whether a music track is AI-generated, where, and why.",
)


@app.get("/health")
def health() -> HealthResponse:
    """Liveness check used by the web app and deployments."""
    return HealthResponse(status="ok", version=__version__)


@app.post("/analyze", responses={501: {"description": "Models are not available yet"}})
async def analyze(
    file: Annotated[UploadFile, File(description="A song or a short recorded clip")],
) -> AnalysisResult:
    """Analyse an uploaded track (Phase 3: calls verasonus.api.inference.analyze_file)."""
    raise HTTPException(
        status_code=501,
        detail="Analysis is not available yet: the models arrive in Phases 2 and 3.",
    )
