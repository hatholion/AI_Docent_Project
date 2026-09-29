"""artifact_id로 유물 정보를 조회하는 로직."""

from __future__ import annotations

import sqlite3

from backend.db.artifacts import get_artifact
from backend.schemas import ArtifactDetail
from backend.services.classification_service import ApiError


def fetch_artifact(connection: sqlite3.Connection, artifact_id: str) -> ArtifactDetail:
    row = get_artifact(connection, artifact_id)
    if row is None:
        raise ApiError(
            404,
            "ARTIFACT_NOT_FOUND",
            f"artifact_id '{artifact_id}'에 해당하는 유물을 찾을 수 없습니다.",
        )
    return ArtifactDetail(**dict(row))
