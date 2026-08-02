"""Repair-generation helpers using the existing SiliconFlow provider."""

from __future__ import annotations

import re

from agent.providers.siliconflow import complete


def clean_source(text: str) -> str:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:cpp|c\+\+|c)?\s*(.*?)\s*```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()
    return value + ("" if value.endswith("\n") else "\n")


def generate_repair(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
):
    """Delegate model access to the existing proven provider."""
    response = complete(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return clean_source(response.content), response
