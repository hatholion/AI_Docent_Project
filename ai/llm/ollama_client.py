"""Minimal client for a local Ollama server."""

from __future__ import annotations

import json
import urllib.request


DEFAULT_HOST = "http://localhost:11434"


def generate(
    prompt: str,
    model: str,
    *,
    host: str = DEFAULT_HOST,
    options: dict | None = None,
) -> dict:
    """Call POST /api/generate (non-streaming) and return the parsed response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))
