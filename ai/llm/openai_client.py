"""Minimal OpenAI Chat Completions client, used only as an external neutral judge.

The docent service itself never calls this - it stays local-LLM-only. This is
purely a development-time evaluation tool (see ai/llm/judge_models_external.py).
"""

from __future__ import annotations

import json
import os
import urllib.request


API_URL = "https://api.openai.com/v1/chat/completions"


def chat(
    prompt: str,
    model: str,
    *,
    api_key: str | None = None,
    temperature: float | None = 0,
    json_mode: bool = True,
) -> str:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Put it in .env (see .env.example) or export it."
        )

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]
