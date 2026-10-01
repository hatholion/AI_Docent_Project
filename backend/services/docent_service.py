"""Business logic for docs/api_spec.md #2/#3/#5: description, chat, chat history.

설명은 사전 생성본(data/rag/gold/emuseum/descriptions_by_type.jsonl, ai/rag/generate_descriptions.py)을
먼저 쓰고, 없으면 ai.rag.docent.Docent가 즉석 생성한다.
채팅 답변은 Docent가 만든다 (기본은 ID 조회, 채팅의 탐색 질문만 RAG 검색).
근거는 data/gold/emuseum/ 카드의 context_text이고, 출처는 카드의 citation이다.
세션·메시지 저장은 backend/db/conversations.py를 그대로 쓴다.
"""

from __future__ import annotations

import sqlite3
import threading
from functools import lru_cache

from ai.rag.docent import ArtifactNotFound, Docent
from ai.rag.gold_store import GoldDataMissing, get_cached_description
from ai.rag.ollama import OllamaError
from backend.db.artifacts import get_artifact
from backend.db.conversations import (
    ArtifactNotFoundError,
    add_message,
    create_conversation,
    get_conversation,
    list_messages,
)
from backend.schemas import (
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DescriptionRequest,
    DescriptionResponse,
)
from backend.services.classification_service import ApiError


CHAT_HISTORY_TURNS = 3  # 최근 N턴(질문+답변)만 프롬프트에 포함

_docent: Docent | None = None
_docent_lock = threading.Lock()


def get_docent() -> Docent:
    """GOLD 카드는 한 번만 읽는다. 검색 색인은 탐색 질문이 올 때 Docent가 연다."""
    global _docent
    with _docent_lock:
        if _docent is None:
            _docent = Docent()
        return _docent


def _artifact_or_404(connection: sqlite3.Connection, artifact_id: str) -> sqlite3.Row:
    artifact = get_artifact(connection, artifact_id)
    if artifact is None:
        raise ApiError(
            404,
            "ARTIFACT_NOT_FOUND",
            f"artifact_id '{artifact_id}'에 해당하는 유물을 찾을 수 없습니다.",
        )
    return artifact


def _pregenerated_description(artifact_id: str, visitor_type: str) -> tuple[str, tuple[str, ...]] | None:
    try:
        text = get_cached_description(artifact_id, visitor_type)
    except GoldDataMissing:
        return None
    docent = get_docent()
    citation = docent.gold.relic_cards[docent.resolve(artifact_id)]["citation"]
    return text, (citation,)


@lru_cache(maxsize=64)
def _cached_description(artifact_id: str, visitor_type: str) -> tuple[str, tuple[str, ...]]:
    # 설명은 유물·관람객 유형마다 같으므로 한 번 만든 것을 재사용한다 (LLM 생성이 수 초 걸림)
    pregenerated = _pregenerated_description(artifact_id, visitor_type)
    if pregenerated is not None:
        return pregenerated
    answer = get_docent().describe(artifact_id, visitor_type=visitor_type)
    return answer.answer, tuple(answer.sources)


def fetch_description(
    connection: sqlite3.Connection, payload: DescriptionRequest
) -> DescriptionResponse:
    artifact = _artifact_or_404(connection, payload.artifact_id)
    try:
        text, sources = _cached_description(payload.artifact_id, payload.visitor_type)
    except ArtifactNotFound as error:
        raise ApiError(404, "DESCRIPTION_NOT_FOUND", str(error)) from error
    except OllamaError as error:
        raise ApiError(503, "LLM_UNAVAILABLE", str(error)) from error

    return DescriptionResponse(
        artifact_id=payload.artifact_id,
        artifact_name=artifact["artifact_name"],
        visitor_type=payload.visitor_type,
        description=text,
        sources=list(sources),
    )


def send_chat_message(connection: sqlite3.Connection, payload: ChatRequest) -> ChatResponse:
    _artifact_or_404(connection, payload.artifact_id)

    if payload.session_id:
        conversation = get_conversation(connection, payload.session_id)
        if conversation is None:
            raise ApiError(
                404,
                "SESSION_NOT_FOUND",
                f"session_id '{payload.session_id}'를 찾을 수 없습니다.",
            )
    else:
        try:
            conversation = create_conversation(
                connection, payload.artifact_id, payload.visitor_type
            )
        except ArtifactNotFoundError as error:
            raise ApiError(
                404,
                "ARTIFACT_NOT_FOUND",
                f"artifact_id '{payload.artifact_id}'에 해당하는 유물을 찾을 수 없습니다.",
            ) from error

    recent = list_messages(connection, conversation.session_id, limit=CHAT_HISTORY_TURNS * 2)
    history = [{"role": message.role, "content": message.content} for message in recent]

    try:
        answer, _route = get_docent().chat(
            payload.artifact_id,
            payload.question,
            visitor_type=payload.visitor_type,
            history=history,
        )
    except ArtifactNotFound as error:
        raise ApiError(404, "ARTIFACT_CONTEXT_NOT_FOUND", str(error)) from error
    except OllamaError as error:
        raise ApiError(503, "LLM_UNAVAILABLE", str(error)) from error

    add_message(connection, conversation.session_id, "user", payload.question)
    add_message(connection, conversation.session_id, "assistant", answer.answer)

    return ChatResponse(
        session_id=conversation.session_id,
        artifact_id=payload.artifact_id,
        visitor_type=payload.visitor_type,
        answer=answer.answer,
        sources=answer.sources,
    )


def fetch_chat_history(connection: sqlite3.Connection, session_id: str) -> ChatHistoryResponse:
    conversation = get_conversation(connection, session_id)
    if conversation is None:
        raise ApiError(404, "SESSION_NOT_FOUND", f"session_id '{session_id}'를 찾을 수 없습니다.")

    messages = list_messages(connection, session_id)
    return ChatHistoryResponse(
        session_id=conversation.session_id,
        artifact_id=conversation.artifact_id,
        visitor_type=conversation.visitor_type,
        messages=[ChatMessage(role=message.role, content=message.content) for message in messages],
    )
