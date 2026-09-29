"""Gold/Chroma/LLM 도슨트 API."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai.llm.chat import make_service
from ai.rag.config import load_config
from ai.rag.gold_loader import RelicNotFoundError
from ai.rag.session import SessionNotFoundError

VisitorType = Literal["child", "general", "expert"]


class DescriptionRequest(BaseModel):
    relic_label: str | None = None
    session_id: str | None = None
    visitor_type: VisitorType | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    relic_label: str | None = None
    visitor_type: VisitorType | None = None


@lru_cache(maxsize=1)
def get_service():
    config_path = os.getenv("DOCENT_CONFIG", "ai/llm/configs/base.yaml")
    return make_service(load_config(config_path))


app = FastAPI(title="Museum AI Docent", version="1.0.0")


def _raise_http(exc: Exception) -> None:
    if isinstance(exc, (RelicNotFoundError, SessionNotFoundError)):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/docent/description")
def describe(request: DescriptionRequest) -> dict:
    try:
        return get_service().describe_relic(
            relic_label=request.relic_label,
            session_id=request.session_id,
            visitor_type=request.visitor_type,
        )
    except Exception as exc:
        _raise_http(exc)


@app.post("/api/v1/docent/chat")
def chat(request: ChatRequest) -> dict:
    try:
        result = get_service().answer_follow_up(
            request.question,
            session_id=request.session_id,
            relic_label=request.relic_label,
            visitor_type=request.visitor_type,
        )
        result["retrieved"] = [
            {
                "chunk_id": item["chunk_id"],
                "relic_label": item["relic_label"],
                "score": item["score"],
                "distance": item["distance"],
            }
            for item in result["retrieved"]
        ]
        return result
    except Exception as exc:
        _raise_http(exc)


@app.get("/api/v1/docent/chat/{session_id}")
def history(session_id: str) -> dict:
    try:
        return get_service().get_session(session_id)
    except Exception as exc:
        _raise_http(exc)
