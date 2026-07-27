#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from pathlib import Path

import requests


SILICONFLOW_URL = "https://api.siliconflow.com/v1/chat/completions"


def strip_code_fence(text: str) -> str:
    """Extract C/C++ source from an optional Markdown code fence."""
    fenced = re.search(
        r"```(?:cpp|c\+\+|c)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced:
        return fenced.group(1).strip() + "\n"

    return text.strip() + "\n"


def generate_candidate(
    prompt: str,
    model: str,
    temperature: float = 0.6,
) -> str:
    api_key = os.environ.get("SILICONFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is not set. "
            "Export it before running the agent."
        )

    print(
        f"Calling model {model} "
        f"with prompt length {len(prompt)} characters; "
        "enable_thinking=False, thinking_budget=128"
    )

    try:
        response = requests.post(
            SILICONFLOW_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert AMD Vitis HLS optimisation agent. "
                                "Return only one complete C++ source file. "
                                "Preserve the function signature and numerical "
                                "behaviour. Do not include explanations or Markdown."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": temperature,
                    "max_tokens": 5000,
                    "enable_thinking": False,
                    "thinking_budget": 128,
                },
            timeout=(30, 600),
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"SiliconFlow request timed out for model {model} "
            f"after waiting up to 600 seconds."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"SiliconFlow request failed before receiving a response: {exc}"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"SiliconFlow returned HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    print(
        f"Model response received: HTTP {response.status_code}"
    )

    payload = response.json()

    try:
        choice = payload["choices"][0]
        message = choice["message"]

        content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason")
        usage = payload.get("usage", {})
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "SiliconFlow returned an unexpected response structure: "
            f"{str(payload)[:2000]}"
        ) from exc

    print(
        "Generation details: "
        f"finish_reason={finish_reason}, "
        f"content_chars={len(content)}, "
        f"reasoning_chars={len(reasoning_content)}, "
        f"usage={usage}"
    )

    if not content.strip():
        raise RuntimeError(
            "The model returned no final content. "
            f"finish_reason={finish_reason}, "
            f"reasoning_chars={len(reasoning_content)}, "
            f"usage={usage}"
        )

    return strip_code_fence(content)


def write_candidate(
    prompt: str,
    model: str,
    output_path: Path,
    temperature: float = 0.2,
) -> None:
    source = generate_candidate(
        prompt=prompt,
        model=model,
        temperature=temperature,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source)
