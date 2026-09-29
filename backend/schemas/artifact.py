"""GET /api/v1/artifacts/{artifact_id} 응답 스키마."""

from __future__ import annotations

from pydantic import BaseModel


class ArtifactDetail(BaseModel):
    artifact_id: str
    artifact_name: str
    source: str
    description: str
    designation_no: str | None = None
