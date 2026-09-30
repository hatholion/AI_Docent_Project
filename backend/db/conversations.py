"""conversations / messages 테이블 스키마 생성과 조회·저장 기능 (설명·채팅 담당).

연결은 backend/db/database.py의 get_connection()을 사용한다.
시각은 UTC ISO 8601 문자열 (예: 2026-09-28T06:12:30.123Z).
"""

from __future__ import annotations

import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import Literal

from backend.db.database import get_connection


VisitorType = Literal["child", "general", "expert"]
Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Conversation:
    session_id: str
    artifact_id: str
    visitor_type: VisitorType
    created_at: str
    last_active_at: str


@dataclass(frozen=True)
class Message:
    id: int
    session_id: str
    role: Role
    content: str
    created_at: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    session_id     TEXT PRIMARY KEY,
    artifact_id    TEXT NOT NULL REFERENCES artifacts (artifact_id),
    visitor_type   TEXT NOT NULL CHECK (visitor_type IN ('child', 'general', 'expert')),
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- 마지막 메시지 시각. 오래된 익명 세션 정리 기준
    last_active_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_conversations_last_active
    ON conversations (last_active_at);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES conversations (session_id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL CHECK (length(content) > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages (session_id, id);
"""


def init_conversations(connection: sqlite3.Connection) -> None:
    """테이블이 없으면 생성한다. artifacts 테이블(init_artifacts)을 먼저 만든 뒤 호출한다."""
    connection.executescript(SCHEMA)


def setup_conversations() -> None:
    """서버 시작 시 호출. 호출한 스레드에서 연결을 열고 닫는다."""
    with closing(get_connection()) as connection:
        init_conversations(connection)


class ArtifactNotFoundError(LookupError):
    """conversations.artifact_id가 artifacts 테이블에 없을 때."""


class SessionNotFoundError(LookupError):
    """존재하지 않는 session_id에 메시지를 추가하려 할 때."""


def new_session_id() -> str:
    # 로그인이 없으므로 session_id를 아는 사람이 곧 대화 주인이다. 추측할 수 없는 값을 쓴다.
    return f"sess_{secrets.token_urlsafe(18)}"


def _to_conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        session_id=row["session_id"],
        artifact_id=row["artifact_id"],
        visitor_type=row["visitor_type"],
        created_at=row["created_at"],
        last_active_at=row["last_active_at"],
    )


def _to_message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
    )


def create_conversation(
    connection: sqlite3.Connection,
    artifact_id: str,
    visitor_type: VisitorType,
) -> Conversation:
    session_id = new_session_id()
    try:
        with connection:
            connection.execute(
                "INSERT INTO conversations (session_id, artifact_id, visitor_type) VALUES (?, ?, ?)",
                (session_id, artifact_id, visitor_type),
            )
    except sqlite3.IntegrityError as error:
        if "FOREIGN KEY" in str(error):
            raise ArtifactNotFoundError(artifact_id) from error
        raise
    conversation = get_conversation(connection, session_id)
    assert conversation is not None
    return conversation


def get_conversation(connection: sqlite3.Connection, session_id: str) -> Conversation | None:
    row = connection.execute(
        "SELECT * FROM conversations WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return _to_conversation(row) if row else None


def add_message(
    connection: sqlite3.Connection,
    session_id: str,
    role: Role,
    content: str,
) -> Message:
    """메시지를 저장하고 세션의 last_active_at을 같은 트랜잭션에서 갱신한다."""
    try:
        with connection:
            cursor = connection.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            connection.execute(
                """
                UPDATE conversations
                SET last_active_at = (SELECT created_at FROM messages WHERE id = ?)
                WHERE session_id = ?
                """,
                (cursor.lastrowid, session_id),
            )
    except sqlite3.IntegrityError as error:
        if "FOREIGN KEY" in str(error):
            raise SessionNotFoundError(session_id) from error
        raise
    row = connection.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _to_message(row)


def list_messages(
    connection: sqlite3.Connection,
    session_id: str,
    *,
    limit: int | None = None,
) -> list[Message]:
    """Messages oldest first. limit을 주면 최근 N개만 (LLM 프롬프트용 대화 맥락)."""
    if limit is None:
        rows = connection.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT * FROM (
                SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?
            ) ORDER BY id
            """,
            (session_id, limit),
        ).fetchall()
    return [_to_message(row) for row in rows]


def delete_inactive_conversations(connection: sqlite3.Connection, *, hours: int = 24) -> int:
    """마지막 활동 후 hours가 지난 세션과 메시지를 삭제하고 삭제한 세션 수를 반환한다."""
    with connection:
        cursor = connection.execute(
            """
            DELETE FROM conversations
            WHERE last_active_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)
            """,
            (f"-{hours} hours",),
        )
    return cursor.rowcount
