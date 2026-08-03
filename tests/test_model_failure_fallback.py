from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.budget import BudgetState
from agent.optimise.runner import run_optimisation
from agent.repair.runner import run_repair


def _repair_config(tmp_path: Path) -> dict[str, object]:
    benchmark = tmp_path / "benchmark"
    (benchmark / "src").mkdir(parents=True)
    (benchmark / "src/kernel.cpp").write_text(
        "void kernel() { /* broken */ }\n",
        encoding="utf-8",
    )
    (benchmark / "src/kernel.h").write_text(
        "void kernel();\n",
        encoding="utf-8",
    )
    (benchmark / "testbench").mkdir()
    (benchmark / "testbench/kernel_tb.cpp").write_text(
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    return {
        "repair_mode": "direct_api",
        "experiment_id": "model_error_retry",
        "benchmark_source": str(benchmark),
        "editable_files": ["src/kernel.cpp"],
        "protected_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "context_files": ["src/kernel.h", "testbench/kernel_tb.cpp"],
        "host_validation": {
            "command": ["true"],
            "run_command": ["true"],
        },
        "independent_validation": {
            "enabled": True,
            "command": ["true"],
        },
        "model": "model",
        "max_attempts": 2,
    }


def _metrics() -> dict[str, int | float]:
    return {
        "clock_period_ns": 2.0,
        "latency_best_cycles": 20,
        "latency_average_cycles": 20,
        "latency_worst_cycles": 20,
        "interval_min_cycles": 1,
        "interval_max_cycles": 1,
        "resources_lut_used": 20,
        "resources_ff_used": 20,
        "resources_dsp_used": 0,
        "resources_bram_used": 0,
    }


def test_repair_model_error_becomes_retryable_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.repair.runner.REPO_ROOT", tmp_path)
    calls = 0

    def fake_generate_repair(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("provider exhausted output budget")
        response = SimpleNamespace(
            content="void kernel() { /* repaired */ }\n",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_seconds=0.1,
            raw_response={},
        )
        return response.content, response

    monkeypatch.setattr("agent.repair.runner.generate_repair", fake_generate_repair)
    budget = BudgetState(
        max_iterations=2,
        max_model_calls=2,
        max_csim_calls=2,
        max_cosim_calls=0,
        max_synthesis_calls=0,
        max_total_tokens=100,
    )

    passed, run_dir, result = run_repair(
        _repair_config(tmp_path),
        keep_workspace=True,
        budget=budget,
    )

    assert passed
    assert result["attempt_count"] == 2
    assert result["attempts"][0]["failed_stage"] == "model_generation"
    assert result["attempts"][0]["failure_class"] == "model_generation_error"
    assert result["attempts"][1]["passed"] is True
    assert budget.model_calls_used == 2
    assert budget.iterations_used == 2
    assert budget.csim_calls_used == 1
    first = run_dir.parent / "attempt_001"
    assert (first / "model_generation_error.log").is_file()
    assert "provider exhausted output budget" in (
        run_dir / "prompt.txt"
    ).read_text(encoding="utf-8")


def test_ppa_model_error_returns_verified_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    monkeypatch.setattr("agent.archive.REPO_ROOT", tmp_path)
    monkeypatch.setattr("agent.optimise.runner._initialise", lambda *args: None)

    def fail_generation(*args, **kwargs):
        raise RuntimeError("provider exhausted output budget")

    monkeypatch.setattr("agent.optimise.runner.generate_candidate", fail_generation)
    (tmp_path / "baseline.cpp").write_text(
        "void kernel() { /* verified baseline */ }\n",
        encoding="utf-8",
    )
    config = {
        "experiment_name": "model_failure_fallback",
        "benchmark": "kernel",
        "top_function": "kernel",
        "baseline": {
            "source": "baseline.cpp",
            "tcl": "task.tcl",
            "project_dir": "baseline_project",
            "metrics": _metrics(),
            "verification": {
                "csim_passed": True,
                "synthesis_passed": True,
                "cosim_passed": True,
            },
        },
        "validation": {},
        "prompt_constraints": [],
        "output_dir": "output",
        "model": {
            "provider": "siliconflow",
            "name": "model",
            "enable_thinking": False,
        },
        "budget": {
            "max_candidates": 1,
            "max_synthesis_calls": 1,
            "max_cosim_calls": 1,
        },
    }

    result = run_optimisation(config, max_steps=1)

    assert result.success is True
    assert result.status == "completed_with_fallback"
    assert result.termination_reason == "model_generation_failed"
    assert [event["stage"] for event in result.trajectory] == [
        "generation",
        "select_best",
    ]
    assert result.trajectory[0]["passed"] is False
    selected = result.trajectory[-1]["selected_design"]
    assert selected["candidate_index"] == 0
    assert selected["fully_verified"] is True
    assert selected["validation"] == {
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": True,
    }
    output = tmp_path / "output"
    assert (output / "candidate_001_generation_error.json").is_file()
    state = json.loads((output / "candidate_state.json").read_text(encoding="utf-8"))
    assert state["selected_design"]["candidate_index"] == 0
    assert state["selected_design_fully_verified"] is True
    assert "verified baseline" in (
        tmp_path / state["selected_design"]["archived_file"]
    ).read_text(encoding="utf-8")
