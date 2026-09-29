"""SQLite 연결을 만드는 공용 모듈.

테이블별 스키마/쿼리 코드는 backend/db/ 아래 각자의 모듈(artifacts.py 등)에 작성하고,
이 모듈의 get_connection()을 가져다 사용한다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ai.vision.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "backend" / "museum.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection
