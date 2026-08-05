from __future__ import annotations

from pathlib import Path

from agent.repair.retry import _record_strategy
from agent.repair.runner import _prompts, run_repair
from agent.repair.strategy import DO_NOT_REPEAT_CONSTRAINT, build_strategy
from agent.state import ValidationResult


def _config(tmp_path: Path) -> dict[str, object]:
    benchmark = tmp_path / "benchmark"
    (benchmark / "src").mkdir(parents=True)
    (benchmark / "src/kernel.cpp").write_text(
        "void kernel(int *out) {\n    out[0] = 0;\n}\n",
        encoding="utf-8",
    )
    (benchmark / "src/kernel.h").write_text(
        "void kernel(int *out);\n",
        encoding="utf-8",
    )
    (benchmark / "testbench").mkdir()
    (benchmark / "testbench/kernel_tb.cpp").write_text(
        "int main() { return 1; }\n",
        encoding="utf-8",
    )
    return {
        "repair_mode": "direct_api",
        "experiment_id": "strategy_test",
        "benchmark_source": str(benchmark),
        "editable_files": ["src/kernel.cpp"],
        "protected_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "context_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "host_validation": {"command": ["true"], "run_command": ["true"]},
        "independent_validation": {"enabled": True, "command": ["true"]},
        "model": "model",
        "max_attempts": 2,
    }


def _failed_result(attempt: int) -> dict[str, object]:
    diagnosis = {
        "stage": "independent_validation",
        "failure_class": "functional_mismatch",
        "summary": "The candidate produced an incorrect value.",
        "suspected_files": ["src/kernel.cpp"],
        "suspected_lines": [],
        "suspected_locations": [],
        "evidence": ["expected=17 actual=16"],
        "repair_constraints": ["Preserve the kernel interface."],
    }
    return {
        "schema_version": 4,
        "experiment_id": "strategy_test",
        "attempt": attempt,
        "repair_mode": "direct_api",
        "provider": "siliconflow",
        "model": "model",
        "failure_class": "functional_mismatch",
        "input_tokens": 10,
        "output_tokens": 5,
        "tokens_used": 15,
        "latency_seconds": 0.1,
        "modified_files": ["src/kernel.cpp"],
        "protected_files_unchanged": True,
        "editable_scope_respected": True,
        "changed_line_count": 2,
        "post_host_validation_passed": False,
        "independent_validation_passed": False,
        "repair_diff_present": True,
        "passed": False,
        "feedback": {
            "attempt": attempt,
            "stage": "independent_validation",
            "failure_class": "functional_mismatch",
            "evidence": ["expected=17 actual=16"],
            "diagnosis": diagnosis,
        },
    }


def test_strategy_is_deterministic_and_records_changed_lines() -> None:
    before = "void kernel(int *out) {\n    out[0] = 0;\n}\n"
    candidate = "void kernel(int *out) {\n    out[0] = 16;\n}\n"

    first = build_strategy(
        before_source=before,
        candidate_source=candidate,
        editable_file="src/kernel.cpp",
    )
    second = build_strategy(
        before_source=before,
        candidate_source=candidate,
        editable_file="src/kernel.cpp",
    )

    assert first["accepted_change"] is True
    assert first["fingerprint"] == second["fingerprint"]
    assert "-     out[0] = 0;" in first["changed_lines"]
    assert "+     out[0] = 16;" in first["changed_lines"]
    assert first["before_hash"] != first["candidate_hash"]


def test_record_strategy_enriches_failure_feedback_and_prompt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    attempt_dir = tmp_path / "attempt_001"
    workspace = attempt_dir / "workspace"
    (workspace / "src").mkdir(parents=True)
    before = "void kernel(int *out) {\n    out[0] = 0;\n}\n"
    candidate = "void kernel(int *out) {\n    out[0] = 16;\n}\n"
    (attempt_dir / "before.cpp").write_text(before, encoding="utf-8")
    (workspace / "src/kernel.cpp").write_text(candidate, encoding="utf-8")

    result = _failed_result(1)
    strategy = _record_strategy(
        attempt_dir=attempt_dir,
        editable="src/kernel.cpp",
        result=result,
    )

    feedback = result["feedback"]
    assert feedback["strategy"] == strategy
    assert any(
        str(item).startswith("Previous strategy:")
        for item in feedback["diagnosis"]["evidence"]
    )
    assert DO_NOT_REPEAT_CONSTRAINT in feedback["diagnosis"]["repair_constraints"]
    assert (attempt_dir / "strategy.json").is_file()

    benchmark = Path(config["benchmark_source"])
    (benchmark / "src/kernel.cpp").write_text(candidate, encoding="utf-8")
    _, prompt = _prompts(
        config,
        benchmark,
        ValidationResult(False, "functional_mismatch", 1, ["current failure"]),
        feedback,
    )

    assert "Previous repair attempt 1 failed" in prompt
    assert "expected=17 actual=16" in prompt
    assert "Previous strategy:" in prompt
    assert strategy["fingerprint"] in prompt
    assert DO_NOT_REPEAT_CONSTRAINT in prompt
    assert candidate in prompt


def test_retry_passes_previous_candidate_stage_evidence_and_strategy(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
                seed_source.read_text(encoding="utf-8") if seed_source else None,
                feedback,
            )
        )
        before = (
            seed_source.read_text(encoding="utf-8")
            if seed_source
            else "void kernel(int *out) {\n    out[0] = 0;\n}\n"
        )
        candidate = (
            "void kernel(int *out) {\n    out[0] = 16;\n}\n"
            if attempt == 1
            else "void kernel(int *out) {\n    out[0] = 17;\n}\n"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "before.cpp").write_text(before, encoding="utf-8")
        path = run_dir / "workspace/src/kernel.cpp"
        path.parent.mkdir(parents=True)
        path.write_text(candidate, encoding="utf-8")

        if attempt == 1:
            return False, run_dir, _failed_result(attempt)

        result = _failed_result(attempt)
        result["passed"] = True
        result["feedback"] = None
        result["post_host_validation_passed"] = True
        result["independent_validation_passed"] = True
        return True, run_dir, result

    monkeypatch.setattr("agent.repair.runner._run_repair_once", fake_attempt)
    passed, _, result = run_repair(config, keep_workspace=True)

    assert passed
    assert result["attempt_count"] == 2
    assert "out[0] = 16" in calls[1][1]
    retry_feedback = calls[1][2]
    assert retry_feedback["stage"] == "independent_validation"
    assert "expected=17 actual=16" in retry_feedback["evidence"]
    assert retry_feedback["strategy"]["accepted_change"] is True
    assert any(
        str(item).startswith("Previous strategy fingerprint:")
        for item in retry_feedback["evidence"]
    )
    assert result["attempts"][0]["strategy"]["fingerprint"]
