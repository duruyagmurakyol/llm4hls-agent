#!/usr/bin/env python3

"""OpenAI-compatible chat-completions transport.

The historical module path is retained for backwards compatibility, but the
transport now supports both SiliconFlow and OpenRouter. Existing callers do
not need to change: select the backend with ``LLM4HLS_PROVIDER``.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from agent.prompt_compaction import compact_user_prompt


DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"
DEFAULT_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
SUPPORTED_PROVIDERS = {"siliconflow", "openrouter"}
NON_THINKING_MODELS = {
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.6-27B",
}
# When thinking is explicitly enabled for source-generation models, cap
# reasoning so it cannot consume the whole completion budget before emitting
# the required source file. Explicit caller-provided thinking_budget wins.
DEFAULT_THINKING_BUDGETS = {
    "Qwen/Qwen3.5-122B-A10B": 1536,
}
STRUCTURED_EXPLORATION_MARKER = "Structured exploration contract:"
CORRECTIVE_RETRY_MARKER = "CORRECTIVE RETRY FOR THIS SAME CANDIDATE SLOT:"
NO_SEMANTIC_CHANGE_MARKER = "because: no_semantic_change."
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ModelResponse:
    content: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_seconds: float
    raw_response: dict[str, Any]
    prompt_compaction: dict[str, Any] | None = None
    provider: str = "siliconflow"


def _provider_name(explicit: str | None = None) -> str:
    value = explicit or os.environ.get("LLM4HLS_PROVIDER", "siliconflow")
    provider = value.strip().casefold()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            "Unsupported LLM provider: "
            f"{value!r}. Expected one of {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )
    return provider


def _api_key(provider: str) -> str:
    names = (
        ("SILICONFLOW_API_KEY", "SILICONFLOW_KEY")
        if provider == "siliconflow"
        else ("OPENROUTER_API_KEY",)
    )
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    expected = names[0]
    raise RuntimeError(
        f"Missing {provider} API key. Export {expected} in your shell."
    )


def _normalise_endpoint(value: str, *, append_chat_path: bool) -> str:
    endpoint = value.rstrip("/")
    if append_chat_path and not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    return endpoint


def _endpoint(provider: str) -> str:
    if provider == "siliconflow":
        base = os.environ.get(
            "SILICONFLOW_BASE_URL",
            DEFAULT_SILICONFLOW_BASE_URL,
        )
        return _normalise_endpoint(base, append_chat_path=True)
    value = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_ENDPOINT)
    return _normalise_endpoint(value, append_chat_path=True)


def _headers(provider: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {_api_key(provider)}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        headers.update(
            {
                "HTTP-Referer": os.environ.get(
                    "OPENROUTER_HTTP_REFERER",
                    "https://llm4hls.local",
                ),
                "X-Title": os.environ.get(
                    "OPENROUTER_X_TITLE",
                    "LLM4HLS Track A",
                ),
            }
        )
    return headers


def _prompt_compaction_enabled(value: bool | None) -> bool:
    if value is not None:
        return value
    configured = os.environ.get("LLM4HLS_COMPACT_PROMPTS", "1")
    return configured.strip().casefold() not in _FALSE_VALUES


def _structured_reasoning_policy(
    *,
    model: str,
    user_prompt: str,
    configured_enable_thinking: bool | None,
) -> tuple[bool | None, str]:
    """Use cheap direct generation first; escalate reasoning only after a no-op.

    Structured C1-C3 prompts already contain a controller-selected strategy.
    Qwen therefore starts without hidden reasoning. A partial strategy
    implementation receives the existing same-slot corrective retry, still
    without thinking. Only a no-semantic-change retry enables thinking.
    Non-structured calls retain their configured behaviour unchanged.
    """

    if (
        model != "Qwen/Qwen3.5-122B-A10B"
        or STRUCTURED_EXPLORATION_MARKER not in user_prompt
    ):
        return configured_enable_thinking, "configured"

    if CORRECTIVE_RETRY_MARKER not in user_prompt:
        return False, "direct_non_thinking"

    if NO_SEMANTIC_CHANGE_MARKER in user_prompt:
        return True, "reasoning_escalation"

    return False, "corrective_non_thinking"


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
    compact_prompt: bool | None = None,
    provider: str | None = None,
) -> ModelResponse:
    """Call the selected OpenAI-compatible provider with bounded retries."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
    if thinking_budget is not None and thinking_budget < 0:
        raise ValueError("thinking_budget must be non-negative")

    selected_provider = _provider_name(provider)
    prompt_compaction: dict[str, Any] | None = None
    if _prompt_compaction_enabled(compact_prompt):
        user_prompt, prompt_compaction = compact_user_prompt(user_prompt)
        saved = int(prompt_compaction["characters_saved"])
        if saved > 0:
            print(
                "Prompt compaction: "
                f"{prompt_compaction['original_characters']} -> "
                f"{prompt_compaction['compacted_characters']} characters "
                f"({prompt_compaction['reduction_percent']:.1f}% reduction)",
                flush=True,
            )

    effective_enable_thinking, generation_mode = _structured_reasoning_policy(
        model=model,
        user_prompt=user_prompt,
        configured_enable_thinking=enable_thinking,
    )
    if effective_enable_thinking is None and model in NON_THINKING_MODELS:
        effective_enable_thinking = False

    effective_thinking_budget = thinking_budget
    if (
        selected_provider == "siliconflow"
        and effective_enable_thinking is True
        and effective_thinking_budget is None
    ):
        effective_thinking_budget = DEFAULT_THINKING_BUDGETS.get(model)

    if generation_mode != "configured":
        print(
            f"Structured generation mode: {generation_mode}; "
            f"thinking={'on' if effective_enable_thinking else 'off'}",
            flush=True,
        )

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
    # SiliconFlow exposes these extension fields. Do not send them to
    # OpenRouter, whose model-specific reasoning controls differ.
    if selected_provider == "siliconflow":
        if effective_thinking_budget is not None:
            payload["thinking_budget"] = effective_thinking_budget
        if effective_enable_thinking is not None:
            payload["enable_thinking"] = effective_enable_thinking

    request_body = json.dumps(payload).encode("utf-8")
    request_url = endpoint or _endpoint(selected_provider)
    overall_started = time.monotonic()
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            request_url,
            data=request_body,
            headers=_headers(selected_provider),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code not in {408, 429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"{selected_provider} HTTP {error.code}: {body}"
                ) from error
            last_error = RuntimeError(
                f"{selected_provider} HTTP {error.code}: {body}"
            )
        except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
            last_error = error

        if attempt == max_attempts:
            raise RuntimeError(
                f"{selected_provider} request failed after {max_attempts} attempts: "
                f"{last_error}"
            ) from last_error

        delay = min(2 ** (attempt - 1), 4)
        print(
            f"{selected_provider} attempt {attempt}/{max_attempts} failed; "
            f"retrying in {delay}s...",
            flush=True,
        )
        time.sleep(delay)
    else:
        raise RuntimeError(f"{selected_provider} request failed without a response")

    latency = time.monotonic() - overall_started
    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError(f"{selected_provider} returned no choices: {raw}")

    message = choices[0].get("message", {})
    content = message.get("content")
    finish_reason = choices[0].get("finish_reason")
    if not isinstance(content, str) or not content.strip():
        reasoning = message.get("reasoning_content")
        if finish_reason == "length" and isinstance(reasoning, str) and reasoning.strip():
            raise RuntimeError(
                f"{selected_provider} exhausted the output budget during reasoning "
                "before returning final content. Set a smaller thinking_budget or "
                "increase max_tokens."
            )
        raise RuntimeError(f"{selected_provider} returned empty content: {raw}")

    raw["_llm4hls_provider"] = selected_provider
    raw["_llm4hls_endpoint"] = request_url
    raw["_llm4hls_generation_mode"] = generation_mode
    raw["_llm4hls_effective_enable_thinking"] = effective_enable_thinking
    if effective_thinking_budget is not None:
        raw["_llm4hls_thinking_budget"] = effective_thinking_budget
    if prompt_compaction is not None:
        raw["_client_prompt_compaction"] = prompt_compaction

    usage = raw.get("usage") or {}
    return ModelResponse(
        content=content,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        latency_seconds=latency,
        raw_response=raw,
        prompt_compaction=prompt_compaction,
        provider=selected_provider,
    )
