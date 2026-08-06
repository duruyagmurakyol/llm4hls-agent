from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import controller
from agent.budget import BudgetExceeded, BudgetState
from agent.config import TaskManifest


def _task(tmp_path: Path, *, requires_cosim: bool) -> TaskManifest:
    source = tmp_path / "kernel.cpp"
    source.write_text("void kernel() {}\n", encoding="utf-8")
    return TaskManifest(
        path=tmp_path / "task.toml",
        data={
            "task_id": "track_a_test",
            "task_kind": "repair",
            "artifacts": {
                "source": str(source),
                "testbench": [str(tmp_path / "tb.cpp")],
                "headers": [],
                "build_files": [str(tmp_path / "task.cfg")],
            },
            "interface": {"top_function": "kernel"},
            "target": {},
            "budgets": {
                "max_iterations": 2,
                "max_model_calls": 2,
                "max_csim_calls": 4,
                "max_cosim_calls": 2,
                "max_synthesis_calls": 4,
                "track_a_credit_budget": 80,
            },
            "model": {},
            "adapter": {"kind": "auto"},
            "output_dir": str(tmp_path / "output"),
            "track_a": {"requires_cosim": requires_cosim},
        },
    )


def test_initial_validation_skips_optional_cosim(tmp_path, monkeypatch) -> None:
    task = _task(tmp_path, requires_cosim=False)
    candidate = Path(task.data["artifacts"]["source"])

    monkeypatch.setattr(
        controller,
        "run_csim",
        lambda _task, _candidate: {
            "passed": True,
            "return_code": 0,
            "timed_out": False,
            "candidate_hash": "abc",
            "candidate_file": str(candidate),
            "log_path": "csim.log",
        },
    )
    monkeypatch.setattr(
        controller,
        "run_synthesis",
        lambda _task, _candidate: {
            "passed": True,
            "return_code": 0,
            "timed_out": False,
            "candidate_hash": "abc",
            "candidate_file": str(candidate),
            "project_dir": str(tmp_path / "project"),
            "log_path": "synth.log",
            "metrics": {"latency_best_cycles": 1},
        },
    )

    def fail_cosim(*_args, **_kwargs):
        raise AssertionError("optional co-simulation was invoked")

    monkeypatch.setattr(controller, "run_cosim", fail_cosim)
    budget = BudgetState.from_manifest(task.data["budgets"])

    route, trajectory, verification = controller._detect_initial_condition(
        task,
        budget,
    )

    assert route == "optimise"
    assert verification is not None
    assert verification["cosim"] is None
    assert budget.cosim_calls_used == 0
    assert budget.track_a_credits_used == 5
    assert [event.stage for event in trajectory] == [
        "initial_csim",
        "initial_synthesis",
        "initial_baseline",
    ]


def test_structural_task_keeps_cosim_required(tmp_path) -> None:
    assert controller._cosim_required(_task(tmp_path, requires_cosim=True)) is True
    assert controller._cosim_required(_task(tmp_path, requires_cosim=False)) is False


def test_credit_exhaustion_retains_existing_verified_baseline(tmp_path) -> None:
    task = _task(tmp_path, requires_cosim=False)
    output_dir = Path(task.output_dir)
    output_dir.mkdir(parents=True)
    selected = output_dir / "active_baseline.cpp"
    selected.write_text("void kernel() {}\n", encoding="utf-8")
    (output_dir / "verified_baseline.json").write_text(
        json.dumps(
            {
                "source": str(selected),
                "validation": {
                    "csim_passed": True,
                    "synthesis_passed": True,
                    "cosim_required": False,
                    "cosim_passed": None,
                },
            }
        ),
        encoding="utf-8",
    )

    budget = BudgetState.from_manifest(
        {
            "max_iterations": 2,
            "max_model_calls": 2,
            "max_csim_calls": 4,
            "max_cosim_calls": 2,
            "max_synthesis_calls": 4,
            "track_a_credit_budget": 1,
        }
    )
    budget.charge_csim(stage="initial_csim")
    with pytest.raises(BudgetExceeded) as captured:
        budget.require("synthesis_calls")

    result = controller._budget_exhausted_result(
        task,
        captured.value,
        budget,
    )

    assert result.success is True
    assert result.status == "verified_baseline"
    assert (
        result.termination_reason
        == "verified_baseline_official_credit_budget_exhausted"
    )
    assert result.to_dict()["selected_design"] == str(selected)
