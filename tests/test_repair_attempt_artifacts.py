from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.budget import BudgetState
from agent.repair.output_validation import InvalidModelOutputError
from agent.repair.runner import run_repair
from agent.state import ValidationResult


BROKEN = """#include \"kernel.h\"
void kernel(const int a[4], int b[4]) {
    for (int i = 0; i < 4; ++i) b[i] = a[i] - 1;
}
"""

REPAIRED = """#include \"kernel.h\"
void kernel(const int a[4], int b[4]) {
    for (int i = 0; i < 4; ++i) b[i] = a[i] + 1;
}
"""

CANONICAL_FILES = {
    "diagnosis.json",
    "system_prompt.txt",
    "prompt.txt",
    "raw_response.txt",
    "candidate.cpp",
    "diff.patch",
    "strategy.json",
    "validation.json",
    "token_usage.json",
    "result.json",
}


def _config(tmp_path: Path) -> dict[str, object]:
    benchmark = tmp_path / "benchmark"
    (benchmark / "src").mkdir(parents=True)
    (benchmark / "testbench").mkdir()
    (benchmark / "src/kernel.cpp").write_text(BROKEN, encoding="utf-8")
    (benchmark / "src/kernel.h").write_text(
        "void kernel(const int a[4], int b[4]);\n",
        encoding="utf-8",
    )
    (benchmark / "testbench/kernel_tb.cpp").write_text(
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    return {
        "repair_mode": "direct_api",
        "experiment_id": "attempt_artifacts",
        "benchmark_source": str(benchmark),
        "editable_files": ["src/kernel.cpp"],
        "protected_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "context_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "host_validation": {"command": ["true"], "run_command": ["true"]},
        "independent_validation": {"enabled": True, "command": ["true"]},
        "model": "model",
        "max_attempts": 1,
    }


def _budget() -> BudgetState:
    return BudgetState(
        max_iterations=1,
        max_model_calls=1,
        max_csim_calls=1,
        max_cosim_calls=0,
        max_synthesis_calls=0,
        max_total_tokens=100,
    )


def _response(content: str = REPAIRED) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        raw_response={"choices": [{"message": {"content": content}}]},
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        latency_seconds=0.1,
    )


def _assert_contract(run_dir: Path) -> tuple[dict, dict, dict]:
    assert run_dir.name == "attempt_001"
    assert CANONICAL_FILES.issubset({path.name for path in run_dir.iterdir()})
    assert (run_dir.parent / "repair_attempts.json").is_file()
    assert (run_dir.parent / "budget_summary.json").is_file()

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    tokens = json.loads((run_dir / "token_usage.json").read_text(encoding="utf-8"))
    assert set(result["artifacts"].values()).issubset(CANONICAL_FILES)
    return result, validation, tokens


def test_successful_repair_writes_complete_attempt_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.repair.runner.REPO_ROOT", tmp_path)
    validations = iter(
        [
            ValidationResult(False, "functional_mismatch", 1, ["expected=1 actual=0"]),
            ValidationResult(True, "none", 0, []),
        ]
    )
    monkeypatch.setattr(
        "agent.repair.runner._validate",
        lambda config, workspace: (next(validations), "validation log\n"),
    )
    monkeypatch.setattr(
        "agent.repair.runner.generate_repair",
        lambda **kwargs: (REPAIRED, _response()),
    )

    passed, run_dir, _ = run_repair(
        _config(tmp_path),
        keep_workspace=True,
        budget=_budget(),
    )

    result, validation, tokens = _assert_contract(run_dir)
    assert passed is True
    assert result["candidate_record_kind"] == "generated_candidate"
    assert (run_dir / "candidate.cpp").read_text(encoding="utf-8") == REPAIRED
    assert validation["model_output_validation"]["passed"] is True
    assert validation["host_after"]["passed"] is True
    assert validation["independent_csim"]["passed"] is True
    assert tokens["total_tokens"] == 15


def test_failed_generated_candidate_records_failed_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.repair.runner.REPO_ROOT", tmp_path)
    validations = iter(
        [
            ValidationResult(False, "functional_mismatch", 1, ["initial mismatch"]),
            ValidationResult(False, "functional_mismatch", 1, ["still mismatched"]),
        ]
    )
    monkeypatch.setattr(
        "agent.repair.runner._validate",
        lambda config, workspace: (next(validations), "validation log\n"),
    )
    monkeypatch.setattr(
        "agent.repair.runner.generate_repair",
        lambda **kwargs: (REPAIRED, _response()),
    )

    passed, run_dir, _ = run_repair(
        _config(tmp_path),
        keep_workspace=True,
        budget=_budget(),
    )

    result, validation, _ = _assert_contract(run_dir)
    assert passed is False
    assert result["candidate_record_kind"] == "generated_candidate"
    assert validation["host_after"]["run"] is True
    assert validation["host_after"]["passed"] is False
    assert (run_dir / "diff.patch").read_text(encoding="utf-8")


def test_rejected_model_output_preserves_last_valid_source_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.repair.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "agent.repair.runner._validate",
        lambda config, workspace: (
            ValidationResult(False, "functional_mismatch", 1, ["initial mismatch"]),
            "validation log\n",
        ),
    )
    response = _response(f"```cpp\n{REPAIRED}```")
    report = {
        "passed": False,
        "failure_class": "invalid_model_output",
        "evidence": ["The response contains a Markdown fence."],
        "violations": ["markdown_fence"],
    }

    def reject(**kwargs):
        raise InvalidModelOutputError(report, response=response)

    monkeypatch.setattr("agent.repair.runner.generate_repair", reject)
    passed, run_dir, _ = run_repair(
        _config(tmp_path),
        keep_workspace=True,
        budget=_budget(),
    )

    result, validation, tokens = _assert_contract(run_dir)
    assert passed is False
    assert result["candidate_record_kind"] == "last_valid_source"
    assert (run_dir / "candidate.cpp").read_text(encoding="utf-8") == BROKEN
    assert validation["model_generation"]["passed"] is True
    assert validation["model_output_validation"]["passed"] is False
    assert validation["host_after"]["run"] is False
    assert tokens["total_tokens"] == 15


def test_provider_failure_still_writes_complete_attempt_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.repair.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "agent.repair.runner._validate",
        lambda config, workspace: (
            ValidationResult(False, "functional_mismatch", 1, ["initial mismatch"]),
            "validation log\n",
        ),
    )

    def fail_provider(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("agent.repair.runner.generate_repair", fail_provider)
    passed, run_dir, _ = run_repair(
        _config(tmp_path),
        keep_workspace=True,
        budget=_budget(),
    )

    result, validation, tokens = _assert_contract(run_dir)
    assert passed is False
    assert result["candidate_record_kind"] == "last_valid_source"
    assert (run_dir / "candidate.cpp").read_text(encoding="utf-8") == BROKEN
    assert (run_dir / "raw_response.txt").read_text(encoding="utf-8") == ""
    assert validation["model_generation"]["passed"] is False
    assert validation["model_output_validation"]["run"] is False
    assert tokens["total_tokens"] == 0
