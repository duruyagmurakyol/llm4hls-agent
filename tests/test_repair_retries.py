from __future__ import annotations

from pathlib import Path

from agent.budget import BudgetState
from agent.repair.runner import _prompts, run_repair
from agent.state import ValidationResult


def _config(tmp_path: Path, *, max_attempts: int | None = 2) -> dict[str, object]:
    benchmark = tmp_path / "benchmark"
    (benchmark / "src").mkdir(parents=True)
    (benchmark / "src/kernel.cpp").write_text("void kernel() { /* broken */ }\n")
    (benchmark / "src/kernel.h").write_text("void kernel();\n")
    (benchmark / "testbench").mkdir()
    (benchmark / "testbench/kernel_tb.cpp").write_text("int main() { return 1; }\n")
    config: dict[str, object] = {
        "repair_mode": "direct_api",
        "experiment_id": "retry_test",
        "benchmark_source": str(benchmark),
        "editable_files": ["src/kernel.cpp"],
        "protected_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "context_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "host_validation": {"command": ["true"], "run_command": ["true"]},
        "independent_validation": {"enabled": True, "command": ["true"]},
        "model": "model",
    }
    if max_attempts is not None:
        config["max_attempts"] = max_attempts
    return config


def _attempt_result(attempt: int, *, passed: bool) -> dict[str, object]:
    return {
        "schema_version": 4,
        "experiment_id": "retry_test",
        "attempt": attempt,
        "repair_mode": "direct_api",
        "provider": "siliconflow",
        "model": "model",
        "thinking_budget": None,
        "failure_class": "functional",
        "pre_host_validation_passed": False,
        "input_tokens": 10,
        "output_tokens": 5,
        "tokens_used": 15,
        "latency_seconds": 0.1,
        "modified_files": ["src/kernel.cpp"],
        "protected_files_unchanged": True,
        "editable_scope_respected": True,
        "changed_line_count": 1,
        "tokens_per_changed_line": 15.0,
        "post_host_validation_passed": passed,
        "independent_validation_passed": passed,
        "repair_diff_present": True,
        "passed": passed,
        "feedback": None,
    }


def test_retry_uses_previous_candidate_and_failure_feedback(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    calls: list[tuple[int, str | None, dict[str, object] | None]] = []

    monkeypatch.setattr("agent.repair.runner.REPO_ROOT", tmp_path)

    def fake_attempt(
        supplied_config,
        *,
        run_dir: Path,
        attempt: int,
        seed_source: Path | None,
        feedback: dict[str, object] | None,
        keep_workspace: bool,
        budget,
    ):
        calls.append(
            (
                attempt,
                seed_source.read_text() if seed_source is not None else None,
                feedback,
            )
        )
        candidate = run_dir / "workspace/src/kernel.cpp"
        candidate.parent.mkdir(parents=True)
        candidate.write_text(f"void kernel() {{ /* attempt {attempt} */ }}\n")
        result = _attempt_result(attempt, passed=attempt == 2)
        failure = {
            "attempt": attempt,
            "stage": "independent_validation",
            "failure_class": "functional",
            "evidence": ["expected 17 actual -15"],
        }
        result["feedback"] = None if attempt == 2 else failure
        return attempt == 2, run_dir, result

    monkeypatch.setattr("agent.repair.runner._run_repair_once", fake_attempt)
    passed, run_dir, result = run_repair(config, keep_workspace=True)

    assert passed
    assert result["attempt_count"] == 2
    assert result["tokens_used"] == 30
    assert result["termination_reason"] == "repair_validated"
    assert calls[0] == (1, None, None)
    assert "attempt 1" in calls[1][1]
    assert calls[1][2]["evidence"] == ["expected 17 actual -15"]
    assert run_dir.name == "attempt_002"
    assert (run_dir / "workspace/src/kernel.cpp").is_file()


def test_prompt_includes_previous_attempt_feedback(tmp_path: Path) -> None:
    config = _config(tmp_path, max_attempts=1)
    workspace = Path(config["benchmark_source"])
    validation = ValidationResult(False, "functional", 1, ["current failure"])

    _, prompt = _prompts(
        config,
        workspace,
        validation,
        {
            "attempt": 1,
            "stage": "independent_validation",
            "failure_class": "functional",
            "evidence": ["expected 17 actual -15"],
        },
    )

    assert "Previous repair attempt 1 failed" in prompt
    assert "expected 17 actual -15" in prompt
    assert "previous candidate" in prompt


def test_retry_stops_when_repair_budget_is_exhausted(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path, max_attempts=None)
    budget = BudgetState(1, 1, 1, 1, 1, 100)
    monkeypatch.setattr("agent.repair.runner.REPO_ROOT", tmp_path)

    def fake_attempt(
        supplied_config,
        *,
        run_dir: Path,
        attempt: int,
        seed_source: Path | None,
        feedback: dict[str, object] | None,
        keep_workspace: bool,
        budget: BudgetState,
    ):
        budget.charge_iteration(stage="repair_attempt_001")
        budget.charge_model_call(stage="repair_attempt_001_generation")
        budget.update_last_event(success=True)
        budget.charge_csim(stage="repair_attempt_001_independent_validation")
        budget.update_last_event(success=False)
        candidate = run_dir / "workspace/src/kernel.cpp"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("void kernel() { /* still broken */ }\n")
        failure = {
            "attempt": 1,
            "stage": "independent_validation",
            "failure_class": "functional",
            "evidence": ["still broken"],
        }
        result = _attempt_result(1, passed=False)
        result["feedback"] = failure
        return False, run_dir, result

    monkeypatch.setattr("agent.repair.runner._run_repair_once", fake_attempt)
    passed, _, result = run_repair(config, keep_workspace=True, budget=budget)

    assert not passed
    assert result["attempt_count"] == 1
    assert result["termination_reason"] == "repair_budget_exhausted"
    assert budget.stop_reason == "repair_budget_exhausted"
