"""Ollama HTTP 클라이언트: 임베딩(BGE-M3)과 채팅(EXAONE). 모델은 config에서 바꾼다."""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np

from ai.rag.config import EMBED_MODEL, LLM_MODEL, OLLAMA_URL


class OllamaError(RuntimeError):
    pass


def _post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        response = httpx.post(f"{OLLAMA_URL}{path}", json=payload, timeout=timeout)
    except httpx.HTTPError as error:
        raise OllamaError(f"Ollama 서버({OLLAMA_URL})에 연결할 수 없습니다: {error}") from error
    if response.status_code != 200:
        raise OllamaError(f"Ollama {path} 실패 ({response.status_code}): {response.text[:300]}")
    return response.json()


def embed(texts: list[str], *, model: str = EMBED_MODEL, batch_size: int = 32) -> np.ndarray:
    """L2 정규화한 임베딩 (len(texts), dim)."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(_post("/api/embed", {"model": model, "input": batch}, timeout=600)["embeddings"])
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = LLM_MODEL,
    temperature: float = 0.2,
    num_ctx: int = 8192,
    max_tokens: int = 700,
    timeout: float = 180.0,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": max_tokens},
    }
    return _post("/api/chat", payload, timeout=timeout)["message"]["content"].strip()


def model_digest(model: str) -> str | None:
    """색인 재현성 기록용 모델 digest (/api/tags)."""
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()
    except httpx.HTTPError:
        return None
    for entry in tags.get("models", []):
        if entry.get("name") in {model, f"{model}:latest"}:
            return entry.get("digest")
    return None
