from __future__ import annotations

import pytest

from agent.providers.siliconflow import ModelResponse
from agent.repair.generate import generate_repair
from agent.repair.prompt import (
    STRICT_REPAIR_SYSTEM_PROMPT,
    build_strict_repair_system_prompt,
)


BASELINE = """#include \"kernel.h\"\n\nvoid kernel(int a[4], int b[4], int c[4]) {\n    for (int i = 0; i < 4; ++i) {\n        c[i] = a[i] - b[i];\n    }\n}\n"""

REPAIRED = """#include \"kernel.h\"\n\nvoid kernel(int a[4], int b[4], int c[4]) {\n    for (int i = 0; i < 4; ++i) {\n        c[i] = a[i] + b[i];\n    }\n}\n"""


def _user_prompt(*, retry: bool = False) -> str:
    prefix = (
        "Previous repair attempt 1 failed.\n"
        "Previous structured diagnosis:\n"
        "Failure class: invalid_model_output\n\n"
        if retry
        else ""
    )
    return (
        prefix
        + "Current structured diagnosis:\n"
        + "Stage: host_validation\n"
        + "Failure class: functional_mismatch\n"
        + "Repair constraints:\n"
        + "- Preserve the kernel function name and signature.\n"
        + "- Modify only: src/kernel.cpp.\n\n"
        + "EDITABLE FILE: src/kernel.cpp\n"
        + "```\n"
        + BASELINE
        + "```\n\n"
        + "FILE: src/kernel.h\n"
        + "```\nvoid kernel(int a[4], int b[4], int c[4]);\n```\n"
    )


def test_strict_prompt_covers_every_fpt_503_requirement() -> None:
    prompt = STRICT_REPAIR_SYSTEM_PROMPT.lower()

    assert "complete contents of the editable source file" in prompt
    assert "raw source text only" in prompt
    assert "markdown fences" in prompt
    assert "explanations" in prompt
    assert "preserve the declared top-function" in prompt
    assert "preserve all behaviour and constraints" in prompt
    assert "smallest justified change" in prompt
    assert "synthesizable amd/xilinx hls c or c++" in prompt
    assert "never modify" in prompt and "testbench" in prompt


def test_additional_instruction_is_subordinate_to_fixed_contract() -> None:
    prompt = build_strict_repair_system_prompt(
        "Return only the complete repaired contents of the editable source file."
    )

    assert prompt.startswith(STRICT_REPAIR_SYSTEM_PROMPT)
    assert "Any additional task instruction is subordinate to these rules." in prompt
    assert "Additional internal task instruction:" in prompt


@pytest.mark.parametrize("retry", [False, True])
def test_generation_uses_same_strict_contract_for_first_attempt_and_retry(
    retry: bool,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_complete(**kwargs):
        captured.update(kwargs)
        return ModelResponse(
            content=REPAIRED,
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            latency_seconds=0.1,
            raw_response={"choices": [{"message": {"content": REPAIRED}}]},
        )

    monkeypatch.setattr("agent.repair.generate.complete", fake_complete)

    repaired, _ = generate_repair(
        model="model",
        system_prompt="Legacy caller guidance.",
        user_prompt=_user_prompt(retry=retry),
    )

    effective = str(captured["system_prompt"])
    assert repaired == REPAIRED
    assert effective.startswith(STRICT_REPAIR_SYSTEM_PROMPT)
    assert "Legacy caller guidance." in effective
    assert captured["enable_thinking"] is False


def test_strict_contract_forbids_test_specific_hard_coding() -> None:
    prompt = STRICT_REPAIR_SYSTEM_PROMPT.lower()

    assert "do not hard-code test vectors" in prompt
    assert "expected outputs" in prompt
    assert "special cases intended only to pass" in prompt
