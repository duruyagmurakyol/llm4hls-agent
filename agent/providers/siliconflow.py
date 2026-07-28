#!/usr/bin/env python3

"""Minimal OpenAI-compatible SiliconFlow chat-completions provider."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "https://api.siliconflow.com/v1"


@dataclass(frozen=True)
class ModelResponse:
    content: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_seconds: float
    raw_response: dict[str, Any]


def _api_key() -> str:
    for name in ("SILICONFLOW_API_KEY", "SILICONFLOW_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        "Missing SiliconFlow API key. Export SILICONFLOW_API_KEY in your shell."
    )


def _endpoint() -> str:
    base_url = os.environ.get("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    return f"{base_url}/chat/completions"


def complete(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_seconds: int = 120,
    endpoint: str | None = None,
) -> ModelResponse:
    """Call SiliconFlow once and return content plus exact usage metadata."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint or _endpoint(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SiliconFlow HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"SiliconFlow request failed: {error}") from error

    latency = time.monotonic() - started
    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError(f"SiliconFlow returned no choices: {raw}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"SiliconFlow returned empty content: {raw}")

    usage = raw.get("usage") or {}
    return ModelResponse(
        content=content,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        latency_seconds=latency,
        raw_response=raw,
    )
