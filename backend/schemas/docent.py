"""Schemas for docs/api_spec.md #2, #3, #5 (description / chat / chat history)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

VisitorType = Literal["child", "general", "expert"]


class DescriptionRequest(BaseModel):
    artifact_id: str
    visitor_type: VisitorType


class DescriptionResponse(BaseModel):
    artifact_id: str
    artifact_name: str
    visitor_type: VisitorType
    description: str
    sources: list[str]


class ChatRequest(BaseModel):
    artifact_id: str
    visitor_type: VisitorType
    session_id: str | None = None
    question: str


class ChatResponse(BaseModel):
    session_id: str
    artifact_id: str
    visitor_type: VisitorType
    answer: str
    sources: list[str]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    artifact_id: str
    visitor_type: VisitorType
    messages: list[ChatMessage]
