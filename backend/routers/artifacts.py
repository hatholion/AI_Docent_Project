"""GET /api/v1/artifacts/{artifact_id} 라우터."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.schemas import ArtifactDetail
from backend.services.artifacts_service import fetch_artifact

router = APIRouter()


@router.get(
    "/api/v1/artifacts/{artifact_id}",
    response_model=ArtifactDetail,
    responses={404: {"description": "Artifact not found"}},
)
async def get_artifact_detail(request: Request, artifact_id: str) -> ArtifactDetail:
    connection = request.app.state.db_connection
    return fetch_artifact(connection, artifact_id)
