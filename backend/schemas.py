"""Public response schemas defined by docs/api_spec.md."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassificationCandidate(BaseModel):
    artifact_id: str
    confidence: float = Field(ge=0, le=1)


class ClassificationSuccess(BaseModel):
    artifact_id: str
    artifact_name: str
    confidence: float = Field(ge=0, le=1)
    candidates: list[ClassificationCandidate]


class ClassificationFailure(BaseModel):
    artifact_id: None = None
    artifact_name: None = None
    confidence: Literal[0.0] = 0.0
    candidates: list[ClassificationCandidate] = Field(default_factory=list)
    message: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
