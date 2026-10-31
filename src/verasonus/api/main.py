"""FastAPI inference service.

Run locally:  uvicorn verasonus.api.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs

Privacy by default: an uploaded file is written to a temporary file and deleted as soon as
its analysis finishes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from verasonus import __version__
from verasonus.api.schemas import AnalysisResult, HealthResponse

app = FastAPI(
    title="VeraSonus API",
    version=__version__,
    description="Estimates whether a music track is AI-generated, where, and why.",
)

MAX_BYTES = 50 * 1024 * 1024  # matches the web app's limit


@app.get("/health")
def health() -> HealthResponse:
    """Liveness check used by the web app and deployments."""
    return HealthResponse(status="ok", version=__version__)


@app.post(
    "/analyze",
    responses={
        413: {"description": "File too large"},
        422: {"description": "Not readable audio, or too short"},
        503: {"description": "Model files are missing"},
    },
)
async def analyze(
    file: Annotated[UploadFile, File(description="A song or a short recorded clip")],
) -> AnalysisResult:
    """Analyse an uploaded track. The upload is deleted as soon as the analysis finishes."""
    from verasonus.api.inference import analyze_file

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Files up to 50 MB.")
    suffix = Path(file.filename or "upload").suffix or ".audio"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"upload{suffix}"
        path.write_bytes(data)
        try:
            return await run_in_threadpool(analyze_file, path)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=f"Cannot analyse: {error}") from error
        except sf.SoundFileError as error:
            detail = "Not readable audio. Try MP3, WAV, FLAC or OGG."
            raise HTTPException(status_code=422, detail=detail) from error
