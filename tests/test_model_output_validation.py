from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.budget import BudgetState
from agent.repair.generate import generate_repair
from agent.repair.output_validation import (
    InvalidModelOutputError,
    validate_model_output,
    validate_response_from_prompt,
)
from agent.repair.runner import run_repair


BASELINE = """#include \"kernel.h\"

void kernel(const int a[16], int b[16]) {
    for (int i = 0; i < 16; ++i) {
        b[i] = a[i] - 1;
    }
}
"""

VALID_REPAIR = """#include \"kernel.h\"

void kernel(const int a[16], int b[16]) {
    for (int i = 0; i < 16; ++i) {
        b[i] = a[i] + 1;
    }
}
"""


def _prompt(source: str = BASELINE) -> str:
    return f"""Current structured diagnosis:
Stage: host_validation
Failure class: functional_mismatch
Repair constraints:
- Preserve the kernel function name and signature.
- Modify only: src/kernel.cpp.

EDITABLE FILE: src/kernel.cpp
```
{source.rstrip()}
```

Return only the full repaired editable file.
"""


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        raw_response={"choices": [{"message": {"content": content}}]},
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        latency_seconds=0.2,
    )


def test_valid_source_passes_all_pre_write_checks() -> None:
    report = validate_model_output(
        raw_response=VALID_REPAIR,
        candidate_source=VALID_REPAIR,
        baseline_source=BASELINE,
        top_function="kernel",
    )

    assert report["passed"] is True
    assert report["violations"] == []
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    ("raw_response", "candidate_source", "expected_violation"),
    [
        ("", "", "empty_response"),
        (f"```cpp\n{VALID_REPAIR}```", VALID_REPAIR, "markdown_fence"),
        ("--- old.cpp\n+++ new.cpp\n@@\n", VALID_REPAIR, "patch_or_multiple_files"),
        (BASELINE, BASELINE, "unchanged_candidate"),
        (VALID_REPAIR.replace("void kernel", "void other"), VALID_REPAIR.replace("void kernel", "void other"), "missing_top_function"),
        (VALID_REPAIR.replace("int b[16]", "long b[16]"), VALID_REPAIR.replace("int b[16]", "long b[16]"), "changed_top_interface"),
        (VALID_REPAIR.rsplit("}", 1)[0], VALID_REPAIR.rsplit("}", 1)[0], "unbalanced_structure"),
    ],
)
def test_required_invalid_shapes_are_rejected(
    raw_response: str,
    candidate_source: str,
    expected_violation: str,
) -> None:
    report = validate_model_output(
        raw_response=raw_response,
        candidate_source=candidate_source,
        baseline_source=BASELINE,
        top_function="kernel",
    )

    assert report["passed"] is False
    assert report["failure_class"] == "invalid_model_output"
    assert expected_violation in report["violations"]


def test_comments_and_string_delimiters_do_not_break_balance_check() -> None:
    candidate = VALID_REPAIR.replace(
        "b[i] = a[i] + 1;",
        'const char *text = "{[(])}"; // unmatched } in comment\n        b[i] = a[i] + 1;',
    )

    report = validate_model_output(
        raw_response=candidate,
        candidate_source=candidate,
        baseline_source=BASELINE,
        top_function="kernel",
    )

    assert report["checks"]["balanced_structure"] is True


def test_prompt_contract_is_used_for_validation() -> None:
    report = validate_response_from_prompt(
        raw_response=VALID_REPAIR,
        candidate_source=VALID_REPAIR,
        user_prompt=_prompt(),
    )

    assert report["passed"] is True
    assert report["expected_top_function"] == "kernel"
    assert report["expected_signature"] == report["candidate_signature"]


def test_generate_repair_rejects_fenced_output(monkeypatch) -> None:
    fenced = f"```cpp\n{VALID_REPAIR}```"
    monkeypatch.setattr(
        "agent.repair.generate.complete",
        lambda **kwargs: _response(fenced),
    )

    with pytest.raises(InvalidModelOutputError) as raised:
        generate_repair(
            model="model",
            system_prompt="Return source only.",
            user_prompt=_prompt(),
        )

    assert "markdown_fence" in raised.value.report["violations"]
    assert raised.value.total_tokens == 15


def _config(tmp_path: Path) -> dict[str, object]:
    benchmark = tmp_path / "benchmark"
    (benchmark / "src").mkdir(parents=True)
    (benchmark / "src/kernel.cpp").write_text(BASELINE)
    (benchmark / "src/kernel.h").write_text(
        "void kernel(const int a[16], int b[16]);\n"
    )
    (benchmark / "testbench").mkdir()
    (benchmark / "testbench/kernel_tb.cpp").write_text("int main() { return 0; }\n")
    return {
        "repair_mode": "direct_api",
        "experiment_id": "invalid_output_test",
        "benchmark_source": str(benchmark),
        "editable_files": ["src/kernel.cpp"],
        "protected_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "context_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "host_validation": {"command": ["true"], "run_command": ["true"]},
        "independent_validation": {"enabled": True, "command": ["true"]},
        "model": "model",
        "max_attempts": 1,
    }


def test_invalid_output_never_overwrites_previous_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    budget = BudgetState(1, 1, 1, 0, 0, 100)
    response = _response(f"```cpp\n{VALID_REPAIR}```")
    report = validate_model_output(
        raw_response=response.content,
        candidate_source=VALID_REPAIR,
        baseline_source=BASELINE,
        top_function="kernel",
    )

    monkeypatch.setattr("agent.repair.runner.REPO_ROOT", tmp_path)

    def reject_output(**kwargs):
        raise InvalidModelOutputError(report, response=response)

    monkeypatch.setattr("agent.repair.runner.generate_repair", reject_output)
    passed, run_dir, result = run_repair(config, keep_workspace=True, budget=budget)

    candidate = run_dir / "workspace/src/kernel.cpp"
    assert passed is False
    assert candidate.read_text() == BASELINE
    assert result["modified_files"] == []
    assert result["feedback"]["failure_class"] == "invalid_model_output"
    assert result["attempts"][0]["candidate_hash"]
    assert result["tokens_used"] == 15
    assert budget.model_calls_used == 1
    assert budget.input_tokens_used == 10
    assert budget.output_tokens_used == 5
    assert budget.csim_calls_used == 0
    assert (run_dir / "output_validation.json").is_file()
    assert (run_dir / "raw_response.txt").read_text() == response.content
    assert not (run_dir / "host_validation_after.log").exists()
