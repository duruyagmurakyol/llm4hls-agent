"""Repair-generation helpers using the SiliconFlow provider."""

from __future__ import annotations

import re

from agent.providers.siliconflow import complete
from agent.repair.output_validation import (
    InvalidModelOutputError,
    validate_response_from_prompt,
)
from agent.repair.prompt import build_strict_repair_system_prompt


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
    timeout_seconds: int = 120,
    thinking_budget: int | None = None,
):
    """Generate and pre-validate one repaired source before it can be written."""
    effective_system_prompt = build_strict_repair_system_prompt(system_prompt)
    response = complete(
        model=model,
        system_prompt=effective_system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        thinking_budget=thinking_budget,
        enable_thinking=False,
    )
    candidate = clean_source(response.content)
    validation = validate_response_from_prompt(
        raw_response=response.content,
        candidate_source=candidate,
        user_prompt=user_prompt,
    )
    if validation["passed"] is not True:
        raise InvalidModelOutputError(validation, response=response)
    return candidate, response
