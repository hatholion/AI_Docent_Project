"""artifacts 테이블 스키마 생성, metadata.csv 시드, 조회 기능."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from ai.vision.config import PROJECT_ROOT

METADATA_CSV_PATH = PROJECT_ROOT / "data" / "metadata.csv"


def create_artifacts_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            artifact_name TEXT NOT NULL,
            source TEXT NOT NULL,
            description TEXT NOT NULL,
            designation_no TEXT
        )
        """
    )
    connection.commit()


def seed_artifacts_from_csv(
    connection: sqlite3.Connection, csv_path: Path = METADATA_CSV_PATH
) -> None:
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = [
            (
                row["artifact_id"],
                row["artifact_name"],
                row["source"],
                row["description"],
            )
            for row in csv.DictReader(csv_file)
        ]
    connection.executemany(
        """
        INSERT INTO artifacts (artifact_id, artifact_name, source, description)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(artifact_id) DO UPDATE SET
            artifact_name = excluded.artifact_name,
            source = excluded.source,
            description = excluded.description
        """,
        rows,
    )
    connection.commit()


def init_artifacts(connection: sqlite3.Connection) -> None:
    """테이블이 없으면 생성하고, 비어있으면 metadata.csv로 시드 데이터를 채운다."""
    create_artifacts_table(connection)
    (count,) = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()
    if count == 0:
        seed_artifacts_from_csv(connection)


def get_artifact(connection: sqlite3.Connection, artifact_id: str) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT artifact_id, artifact_name, source, description, designation_no
        FROM artifacts
        WHERE artifact_id = ?
        """,
        (artifact_id,),
    ).fetchone()
