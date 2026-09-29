"""개발 단계용 thread-safe in-memory 대화 세션."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4


class SessionNotFoundError(LookupError):
    pass


@dataclass
class ChatSession:
    session_id: str
    current_relic_label: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = RLock()

    def create(self, *, session_id: str | None = None,
               current_relic_label: str | None = None) -> ChatSession:
        identifier = session_id.strip() if isinstance(session_id, str) and session_id.strip() else f"sess_{uuid4().hex}"
        with self._lock:
            session = self._sessions.get(identifier)
            if session is None:
                session = ChatSession(identifier, current_relic_label)
                self._sessions[identifier] = session
            elif current_relic_label is not None:
                session.current_relic_label = current_relic_label
            return deepcopy(session)

    def get(self, session_id: str) -> ChatSession:
        with self._lock:
            try:
                return deepcopy(self._sessions[session_id])
            except KeyError as exc:
                raise SessionNotFoundError(f"대화 세션을 찾을 수 없습니다: {session_id}") from exc

    def set_current_relic(self, session_id: str, relic_label: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(f"대화 세션을 찾을 수 없습니다: {session_id}")
            self._sessions[session_id].current_relic_label = relic_label

    def append(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("history role은 user 또는 assistant여야 합니다")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history content는 비어 있을 수 없습니다")
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(f"대화 세션을 찾을 수 없습니다: {session_id}")
            self._sessions[session_id].history.append({"role": role, "content": content.strip()})

    def recent_history(self, session_id: str, turns: int) -> list[dict[str, str]]:
        history = self.get(session_id).history
        return history[-(turns * 2):] if turns > 0 else []
