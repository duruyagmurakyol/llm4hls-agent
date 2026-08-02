#!/usr/bin/env python3

"""Minimal OpenAI-compatible SiliconFlow chat-completions provider."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "https://api.siliconflow.com/v1"
NON_THINKING_MODELS = {
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.6-27B",
}


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
    timeout_seconds: int = 180,
    endpoint: str | None = None,
    max_attempts: int = 3,
    thinking_budget: int | None = None,
    enable_thinking: bool | None = None,
) -> ModelResponse:
    """Call SiliconFlow and retry transient network/read failures."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
    if thinking_budget is not None and thinking_budget < 0:
        raise ValueError("thinking_budget must be non-negative")

    effective_enable_thinking = enable_thinking
    if effective_enable_thinking is None and model in NON_THINKING_MODELS:
        effective_enable_thinking = False

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if thinking_budget is not None and effective_enable_thinking is not False:
        payload["thinking_budget"] = thinking_budget
    if effective_enable_thinking is not None:
        payload["enable_thinking"] = effective_enable_thinking

    request_body = json.dumps(payload).encode("utf-8")
    request_url = endpoint or _endpoint()
    overall_started = time.monotonic()
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            request_url,
            data=request_body,
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code not in {408, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"SiliconFlow HTTP {error.code}: {body}") from error
            last_error = RuntimeError(f"SiliconFlow HTTP {error.code}: {body}")
        except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
            last_error = error

        if attempt == max_attempts:
            raise RuntimeError(
                f"SiliconFlow request failed after {max_attempts} attempts: {last_error}"
            ) from last_error

        delay = min(2 ** (attempt - 1), 4)
        print(
            f"SiliconFlow attempt {attempt}/{max_attempts} failed; retrying in {delay}s...",
            flush=True,
        )
        time.sleep(delay)
    else:
        raise RuntimeError("SiliconFlow request failed without a response")

    latency = time.monotonic() - overall_started
    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError(f"SiliconFlow returned no choices: {raw}")

    message = choices[0].get("message", {})
    content = message.get("content")
    finish_reason = choices[0].get("finish_reason")
    if not isinstance(content, str) or not content.strip():
        reasoning = message.get("reasoning_content")
        if finish_reason == "length" and isinstance(reasoning, str) and reasoning.strip():
            raise RuntimeError(
                "SiliconFlow exhausted the output budget during reasoning before returning "
                "final content. Disable thinking for this experiment or increase max_tokens."
            )
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
