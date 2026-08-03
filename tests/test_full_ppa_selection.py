from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.budget import BudgetState
from agent.optimise.evaluate import classify_candidate
from agent.optimise.metrics import derive_hardware_metrics
from agent.optimise.runner import run_optimisation
from agent.optimise.selection import deterministic_selection_key


def _metrics(
    *,
    clock: float = 2.0,
    latency: int = 10,
    interval: int = 1,
    lut: int = 20,
    ff: int = 20,
    dsp: int = 0,
    bram: int = 0,
) -> dict[str, int | float]:
    return {
        "clock_period_ns": clock,
        "latency_best_cycles": latency,
        "latency_average_cycles": latency,
        "latency_worst_cycles": latency,
        "interval_min_cycles": interval,
        "interval_max_cycles": interval,
        "resources_lut_used": lut,
        "resources_ff_used": ff,
        "resources_dsp_used": dsp,
        "resources_bram_used": bram,
    }


def _candidate_files(
    output: Path,
    *,
    metrics: dict[str, int | float],
    cosim: bool | None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidate_001.cpp").write_text("void kernel() {}\n", encoding="utf-8")
    (output / "candidate_001_static_validation.json").write_text(
        json.dumps({"passed": True, "checks": {"source": True}}),
        encoding="utf-8",
    )
    (output / "candidate_001_csim_validation.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    (output / "candidate_001_synthesis.json").write_text(
        json.dumps(
            {
                "passed": True,
                "synthesis_run": True,
                "timed_out": False,
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    if cosim is not None:
        (output / "candidate_001_cosim.json").write_text(
            json.dumps(
                {
                    "passed": cosim,
                    "cosim_run": True,
                    "timed_out": False,
                }
            ),
            encoding="utf-8",
        )


def test_candidate_waits_for_cosim_after_synthesis(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    baseline = derive_hardware_metrics(_metrics(latency=20), minimum_frequency_mhz=100)
    output = tmp_path / "out"
    _candidate_files(output, metrics=_metrics(latency=10), cosim=None)

    record = classify_candidate(
        output,
        1,
        baseline,
        {},
        minimum_frequency_mhz=100,
    )

    assert record["verdict"] == "awaiting_cosim"
    assert record["fully_verified"] is False
    assert record["synthesis"] is True
    assert record["cosim"] is None


def test_resource_limit_violation_is_rejected_before_cosim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    baseline = derive_hardware_metrics(_metrics(latency=20), minimum_frequency_mhz=100)
    output = tmp_path / "out"
    _candidate_files(output, metrics=_metrics(latency=8, lut=120), cosim=None)

    record = classify_candidate(
        output,
        1,
        baseline,
        {},
        minimum_frequency_mhz=100,
        resource_limits={"lut": 100},
    )

    assert record["verdict"] == "reject_resource_limits"
    assert record["meets_resource_limits"] is False
    assert record["cosim_run"] is False
    assert record["resource_limit_compliance"]["violations"][0]["metric"] == "resources_lut_used"


def test_failed_cosim_can_never_enter_selection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    baseline = derive_hardware_metrics(_metrics(latency=20), minimum_frequency_mhz=100)
    output = tmp_path / "out"
    _candidate_files(output, metrics=_metrics(latency=8), cosim=False)

    record = classify_candidate(
        output,
        1,
        baseline,
        {},
        minimum_frequency_mhz=100,
    )

    assert record["verdict"] == "reject_cosim"
    assert record["fully_verified"] is False


def test_deterministic_ranking_uses_cost_then_candidate_index() -> None:
    metrics = derive_hardware_metrics(_metrics(latency=8), minimum_frequency_mhz=100)

    def record(index: int, tokens: int) -> dict[str, Any]:
        return {
            "candidate_index": index,
            "fully_verified": True,
            "meets_frequency_requirement": True,
            "resource_limit_compliance": {"passed": True},
            "metrics": metrics,
            "cost": {
                "total_tokens": tokens,
                "tool_calls": 3,
                "tool_seconds": 1.0,
            },
        }

    expensive = record(1, 100)
    cheaper = record(2, 50)
    identical_lower_index = record(1, 50)

    assert deterministic_selection_key(cheaper) < deterministic_selection_key(expensive)
    assert deterministic_selection_key(identical_lower_index) < deterministic_selection_key(cheaper)


def test_optimiser_runs_cosim_before_selecting_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    monkeypatch.setattr("agent.archive.REPO_ROOT", tmp_path)
    monkeypatch.setattr("agent.optimise.runner._initialise", lambda *args: None)

    baseline = tmp_path / "baseline.cpp"
    baseline.write_text("void kernel() { /* baseline */ }\n", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()

    config = {
        "experiment_name": "full_selection",
        "benchmark": "kernel",
        "top_function": "kernel",
        "minimum_frequency_mhz": 100.0,
        "resource_limits": {"lut": 100},
        "selection": {},
        "baseline": {
            "source": "baseline.cpp",
            "tcl": "task.cfg",
            "project_dir": "baseline_project",
            "metrics": _metrics(latency=20, lut=30),
            "verification": {
                "csim_passed": True,
                "synthesis_passed": True,
                "cosim_passed": True,
            },
        },
        "validation": {},
        "prompt_constraints": [],
        "output_dir": "out",
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

    def fake_generate(config_source, index: int, *, budget=None):
        candidate = output / f"candidate_{index:03d}.cpp"
        candidate.write_text("void kernel() { /* faster */ }\n", encoding="utf-8")
        (output / f"candidate_{index:03d}_model_metadata.json").write_text(
            json.dumps(
                {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                }
            ),
            encoding="utf-8",
        )
        return candidate

    def fake_static(config_source, index: int):
        report = {"passed": True, "checks": {"source": True}}
        (output / f"candidate_{index:03d}_static_validation.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    def fake_csim(config_source, index: int):
        report = {"passed": True, "return_code": 0, "timed_out": False, "elapsed_seconds": 0.1}
        (output / f"candidate_{index:03d}_csim_validation.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    def fake_synthesis(config_source, index: int):
        report = {
            "passed": True,
            "return_code": 0,
            "timed_out": False,
            "elapsed_seconds": 0.2,
            "synthesis_run": True,
            "metrics": _metrics(latency=8, lut=25),
        }
        (output / f"candidate_{index:03d}_synthesis.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    def fake_cosim(config_source, index: int):
        report = {
            "passed": True,
            "return_code": 0,
            "timed_out": False,
            "elapsed_seconds": 0.3,
            "cosim_run": True,
        }
        (output / f"candidate_{index:03d}_cosim.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    monkeypatch.setattr("agent.optimise.runner.generate_candidate", fake_generate)
    monkeypatch.setattr("agent.optimise.runner.validate_ppa_candidate", fake_static)
    monkeypatch.setattr(
        "agent.optimise.runner.check_candidate_duplicate",
        lambda config_source, index: {"passed": True, "duplicate_of": None},
    )
    monkeypatch.setattr("agent.optimise.runner.run_candidate_csim", fake_csim)
    monkeypatch.setattr("agent.optimise.runner.run_candidate_synthesis", fake_synthesis)
    monkeypatch.setattr("agent.optimise.runner.run_candidate_cosim", fake_cosim)

    budget = BudgetState(
        max_iterations=1,
        max_model_calls=1,
        max_csim_calls=1,
        max_cosim_calls=1,
        max_synthesis_calls=1,
        max_total_tokens=100,
    )
    result = run_optimisation(config, max_steps=1, budget=budget)

    assert result.success is True
    assert result.termination_reason == "candidate_dominates_baseline"
    assert [event["stage"] for event in result.trajectory] == [
        "generation",
        "static_validation",
        "duplicate_check",
        "csim",
        "synthesis",
        "cosim",
        "select_best",
    ]
    assert result.summary["selected_design"]["candidate_index"] == 1
    assert result.summary["selected_design_fully_verified"] is True
    assert result.summary["selected_design_resource_compliant"] is True
    assert budget.csim_calls_used == 1
    assert budget.synthesis_calls_used == 1
    assert budget.cosim_calls_used == 1
