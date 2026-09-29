"""Routers for POST /docent/description, POST /docent/chat, GET /docent/chat/{id}
(docs/api_spec.md #2, #3, #5).
"""

from __future__ import annotations

from contextlib import closing

from fastapi import APIRouter

from backend.db.database import get_connection
from backend.schemas import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    DescriptionRequest,
    DescriptionResponse,
)
from backend.services.docent_service import fetch_chat_history, fetch_description, send_chat_message

router = APIRouter()


@router.post(
    "/api/v1/docent/description",
    response_model=DescriptionResponse,
    responses={404: {"description": "Artifact or description not found"}},
)
def describe_artifact(payload: DescriptionRequest) -> DescriptionResponse:
    # def 라우트는 작업 스레드에서 실행되므로, 같은 스레드에서 연결을 열고 닫는다
    with closing(get_connection()) as connection:
        return fetch_description(connection, payload)


@router.post(
    "/api/v1/docent/chat",
    response_model=ChatResponse,
    responses={404: {"description": "Artifact or session not found"}},
)
def chat(payload: ChatRequest) -> ChatResponse:
    with closing(get_connection()) as connection:
        return send_chat_message(connection, payload)


@router.get(
    "/api/v1/docent/chat/{session_id}",
    response_model=ChatHistoryResponse,
    responses={404: {"description": "Session not found"}},
)
def get_chat_history(session_id: str) -> ChatHistoryResponse:
    with closing(get_connection()) as connection:
        return fetch_chat_history(connection, session_id)
