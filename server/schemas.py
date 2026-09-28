"""Pydantic models for the playground API."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Primitive = Literal["noul", "score", "choice"]
ModelChoice = Literal["auto", "english", "multilingual"]

PRIMITIVE_TO_KEY: dict[str, str] = {
    "noul": "is_scam",
    "score": "risk_level",
    "choice": "scam_category",
}


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")
    primitives: list[Primitive] = Field(..., min_length=1)
    model: ModelChoice = "auto"
    questions: Optional[dict[str, Any]] = None
    max_len: Optional[int] = None


class PredictResponse(BaseModel):
    answers: dict[str, Any]
    routing: dict[str, Any]
    latency_ms: float


class HealthResponse(BaseModel):
    ok: bool
    models: dict[str, bool]
    multilingual_dir: str
    load_error: Optional[str] = None


class Sample(BaseModel):
    id: str
    label: str
    text: str
    lang: str


class SamplesResponse(BaseModel):
    samples: list[Sample]