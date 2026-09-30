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
            accession_no TEXT,
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
                (row.get("accession_no") or "").strip() or None,
                row["artifact_name"],
                row["source"],
                row["description"],
            )
            for row in csv.DictReader(csv_file)
        ]
    connection.executemany(
        """
        INSERT INTO artifacts (artifact_id, accession_no, artifact_name, source, description)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(artifact_id) DO UPDATE SET
            accession_no = excluded.accession_no,
            artifact_name = excluded.artifact_name,
            source = excluded.source,
            description = excluded.description
        """,
        rows,
    )
    connection.commit()


def add_missing_columns(connection: sqlite3.Connection) -> bool:
    """이전 버전으로 만든 DB에 없는 컬럼을 추가한다. 추가했으면 True."""
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(artifacts)")}
    if "accession_no" in columns:
        return False
    connection.execute("ALTER TABLE artifacts ADD COLUMN accession_no TEXT")
    connection.commit()
    return True


def init_artifacts(connection: sqlite3.Connection) -> None:
    """테이블이 없으면 생성하고, 비어있으면 metadata.csv로 시드 데이터를 채운다."""
    create_artifacts_table(connection)
    added_columns = add_missing_columns(connection)
    (count,) = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()
    # 새로 추가한 컬럼은 비어 있으므로 기존 행도 metadata.csv 값으로 다시 채운다
    if count == 0 or added_columns:
        seed_artifacts_from_csv(connection)


def get_artifact(connection: sqlite3.Connection, artifact_id: str) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT artifact_id, accession_no, artifact_name, source, description, designation_no
        FROM artifacts
        WHERE artifact_id = ?
        """,
        (artifact_id,),
    ).fetchone()
