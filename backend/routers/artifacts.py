"""GET /api/v1/artifacts/{artifact_id} 라우터."""

from __future__ import annotations

from contextlib import closing

from fastapi import APIRouter

from backend.db.database import get_connection
from backend.schemas import ArtifactDetail
from backend.services.artifacts_service import fetch_artifact

router = APIRouter()


@router.get(
    "/api/v1/artifacts/{artifact_id}",
    response_model=ArtifactDetail,
    responses={404: {"description": "Artifact not found"}},
)
def get_artifact_detail(artifact_id: str) -> ArtifactDetail:
    # def 라우트는 작업 스레드에서 실행되므로, 같은 스레드에서 연결을 열고 닫는다
    with closing(get_connection()) as connection:
        return fetch_artifact(connection, artifact_id)
