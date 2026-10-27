"""API response schemas. web/src/lib/api.ts mirrors these types: keep them in sync."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Segment(BaseModel):
    start_seconds: float
    end_seconds: float
    ai_probability: float = Field(ge=0, le=1)


class GeneratorGuess(BaseModel):
    name: str  # e.g. "suno"; "unknown" when the outlier detector fires
    probability: float = Field(ge=0, le=1)


class FeatureContribution(BaseModel):
    name: str
    contribution: float  # positive pushes towards "AI", negative towards "real"


class Explanation(BaseModel):
    summary: str
    top_features: list[FeatureContribution] = []


class AnalysisResult(BaseModel):
    ai_probability: float = Field(
        ge=0, le=1, description="Calibrated probability that the track is AI-generated"
    )
    verdict: Literal["likely_ai", "uncertain", "likely_human"]
    confidence: float = Field(ge=0, le=1)
    duration_seconds: float
    segments: list[Segment]
    generator: GeneratorGuess | None = None
    explanation: Explanation
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
