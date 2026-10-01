"""Smoke-test the artifacts / conversations / messages SQLite tables on a temporary DB."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.artifacts import get_artifact, init_artifacts  # noqa: E402
from backend.db.database import get_connection  # noqa: E402
from backend.db.conversations import (  # noqa: E402
    ArtifactNotFoundError,
    SessionNotFoundError,
    add_message,
    create_conversation,
    delete_inactive_conversations,
    get_conversation,
    init_conversations,
    list_messages,
)



def check(condition: bool, label: str) -> None:
    print(f"[{'OK' if condition else 'FAIL'}] {label}")
    if not condition:
        raise SystemExit(1)


def check_legacy_artifacts_upgrade(db_path: Path) -> None:
    """accession_no 컬럼 없이 만들어진 기존 DB도 서버 시작 시 채워지는지."""
    with closing(get_connection(db_path)) as connection:
        connection.execute(
            """
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY, artifact_name TEXT NOT NULL,
                source TEXT NOT NULL, description TEXT NOT NULL, designation_no TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO artifacts VALUES ('bon002789', '금동 반가사유상', '국립중앙박물관', 'TODO', NULL)"
        )
        connection.commit()
        init_artifacts(connection)
        row = get_artifact(connection, "bon002789")
    check(row is not None and row["accession_no"] == "본관 2789", "legacy DB gets accession_no")


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        check_legacy_artifacts_upgrade(Path(temp_dir) / "legacy.db")

        with closing(get_connection(Path(temp_dir) / "check.db")) as connection:
            init_artifacts(connection)
            init_conversations(connection)
            init_conversations(connection)
            check(True, "init can run twice")
            artifact = get_artifact(connection, "ssu022891")
            check(artifact is not None and artifact["accession_no"] == "신수 22891", "accession_no seeded from metadata.csv")

            session_id = create_conversation(connection, "bon002789", "general").session_id
            check(session_id.startswith("sess_") and len(session_id) > 20, f"session id {session_id}")

            add_message(connection, session_id, "user", "왜 반가사유상이라고 불러요?")
            reply = add_message(connection, session_id, "assistant", "반가부좌 자세로 사유하는 모습에서...")
            add_message(connection, session_id, "user", "언제 만들어졌어요?")
            stored = get_conversation(connection, session_id)
            check(stored is not None and stored.last_active_at >= reply.created_at, "last_active_at updated")

            messages = list_messages(connection, session_id)
            recent = list_messages(connection, session_id, limit=2)
            check([m.role for m in messages] == ["user", "assistant", "user"], "messages in order")
            check([m.id for m in recent] == [m.id for m in messages[-2:]], "limit keeps latest, oldest first")

            check(get_conversation(connection, "sess_missing") is None, "unknown session -> None")
            try:
                create_conversation(connection, "no_such_artifact", "child")
                check(False, "unknown artifact rejected")
            except ArtifactNotFoundError:
                check(True, "unknown artifact rejected")
            try:
                add_message(connection, "sess_missing", "user", "hi")
                check(False, "message to unknown session rejected")
            except SessionNotFoundError:
                check(True, "message to unknown session rejected")
            for label, sql, params in [
                ("invalid visitor_type rejected",
                 "INSERT INTO conversations (session_id, artifact_id, visitor_type) VALUES (?, ?, ?)",
                 ("sess_x", "bon002789", "adult")),
                ("invalid role rejected",
                 "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                 (session_id, "system", "x")),
                ("empty content rejected",
                 "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                 (session_id, "user", "")),
            ]:
                try:
                    with connection:
                        connection.execute(sql, params)
                    check(False, label)
                except sqlite3.IntegrityError:
                    check(True, label)

            fresh = create_conversation(connection, "bon002789", "child").session_id
            with connection:
                connection.execute(
                    "UPDATE conversations SET last_active_at = '2000-01-01T00:00:00.000Z' WHERE session_id = ?",
                    (session_id,),
                )
            deleted = delete_inactive_conversations(connection, hours=24)
            orphan_messages = connection.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            check(deleted == 1 and get_conversation(connection, fresh) is not None, "only inactive session deleted")
            check(orphan_messages == 0, "messages cascade-deleted")

    print("docent DB check passed")


if __name__ == "__main__":
    main()
