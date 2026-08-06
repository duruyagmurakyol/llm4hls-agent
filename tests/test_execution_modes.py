from __future__ import annotations

from pathlib import Path

import pytest

from agent import controller
from agent.budget import BudgetState
from agent.config import TaskManifest
from agent.execution_mode import _run_optimise_only, _run_repair_only
from agent.state import AgentResult, TrajectoryEvent


def _task(tmp_path: Path) -> TaskManifest:
    return TaskManifest(
        path=tmp_path / "task.json",
        data={
            "task_id": "mode_test",
            "task_kind": "repair",
            "output_dir": str(tmp_path / "output"),
            "target": {"minimum_frequency_mhz": 100.0},
            "track_a": {"requires_cosim": False},
            "adapter": {"kind": "auto"},
        },
    )


def _budget() -> BudgetState:
    return BudgetState(
        max_iterations=4,
        max_model_calls=4,
        max_csim_calls=8,
        max_cosim_calls=4,
        max_synthesis_calls=8,
        requires_cosim=False,
    )


def _baseline(origin: str) -> dict[str, object]:
    return {
        "origin": origin,
        "source": "experiments/example/active_baseline.cpp",
        "candidate_hash": "abc123",
        "project_dir": "experiments/example/verified_baseline_project",
        "metrics": {
            "clock_period_ns": 5.0,
            "frequency_mhz": 200.0,
            "latency_worst_cycles": 32,
        },
        "validation": {
            "csim_passed": True,
            "synthesis_passed": True,
            "cosim_required": False,
            "cosim_passed": None,
        },
    }


def _failed_initial(stage: str = "initial_csim") -> list[TrajectoryEvent]:
    return [
        TrajectoryEvent(
            step=1,
            stage=stage,
            status="failed",
            details={"failure_class": "functional_mismatch"},
        )
    ]


def test_repair_mode_stops_after_verified_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    repaired = AgentResult(
        task_id=task.task_id,
        success=True,
        status="correctness_and_synthesis_established",
        termination_reason="repair_and_synthesis_completed",
        output_dir=str(task.output_dir),
        trajectory=[
            TrajectoryEvent(step=1, stage="repair", status="passed", details={})
        ],
    )
    monkeypatch.setattr(
        controller,
        "_detect_initial_condition",
        lambda *_: ("repair", _failed_initial(), None),
    )
    monkeypatch.setattr(controller, "_run_direct_api_repair", lambda *_: repaired)
    monkeypatch.setattr(
        controller,
        "_repair_baseline",
        lambda *_: _baseline("repaired"),
    )
    monkeypatch.setattr(
        controller,
        "_run_autonomous_ppa",
        lambda *args, **kwargs: pytest.fail("repair mode entered PPA optimisation"),
    )

    result = _run_repair_only(
        task,
        budget=_budget(),
        status_only=False,
    )

    assert result.success is True
    assert result.termination_reason == "repair_only_verified"
    assert any(event.stage == "repair" for event in result.trajectory)
    assert not any(event.stage == "generation" for event in result.trajectory)


def test_repair_mode_skips_model_when_source_is_already_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    verification = {"source": Path("source.cpp")}
    monkeypatch.setattr(
        controller,
        "_detect_initial_condition",
        lambda *_: ("optimise", [], verification),
    )
    monkeypatch.setattr(
        controller,
        "_initial_baseline",
        lambda *_: _baseline("initial"),
    )
    monkeypatch.setattr(
        controller,
        "_run_direct_api_repair",
        lambda *_: pytest.fail("already verified source triggered a model repair"),
    )
    monkeypatch.setattr(
        controller,
        "_run_autonomous_ppa",
        lambda *args, **kwargs: pytest.fail("repair mode entered PPA optimisation"),
    )

    result = _run_repair_only(
        task,
        budget=_budget(),
        status_only=False,
    )

    assert result.success is True
    assert result.termination_reason == "repair_only_already_verified"
    assert (tmp_path / "output" / "candidate_state.json").is_file()


def test_optimise_mode_rejects_invalid_baseline_without_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    monkeypatch.setattr(
        controller,
        "_detect_initial_condition",
        lambda *_: ("repair", _failed_initial("initial_synthesis"), None),
    )
    monkeypatch.setattr(
        controller,
        "_run_direct_api_repair",
        lambda *_: pytest.fail("optimise mode attempted to repair an invalid baseline"),
    )
    monkeypatch.setattr(
        controller,
        "_run_autonomous_ppa",
        lambda *args, **kwargs: pytest.fail("invalid baseline entered PPA optimisation"),
    )

    result = _run_optimise_only(
        task,
        budget=_budget(),
        status_only=False,
        max_steps=None,
    )

    assert result.success is False
    assert result.status == "invalid_optimisation_baseline"
    assert result.termination_reason == "optimisation_baseline_invalid"
    guard = next(event for event in result.trajectory if event.stage == "mode_guard")
    assert guard.details["failed_stage"] == "initial_synthesis"
    assert guard.details["action"] == "rejected_without_repair"


def test_optimise_mode_promotes_valid_initial_baseline_then_runs_ppa(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    verification = {"source": Path("source.cpp")}
    called: dict[str, object] = {}
    monkeypatch.setattr(
        controller,
        "_detect_initial_condition",
        lambda *_: ("optimise", [], verification),
    )
    monkeypatch.setattr(
        controller,
        "_initial_baseline",
        lambda *_: _baseline("initial"),
    )
    monkeypatch.setattr(
        controller,
        "_run_direct_api_repair",
        lambda *_: pytest.fail("optimise mode called the repair workflow"),
    )

    def fake_ppa(*args: object, **kwargs: object) -> AgentResult:
        called["baseline"] = kwargs["baseline"]
        return AgentResult(
            task_id=task.task_id,
            success=True,
            status="completed",
            termination_reason="optimisation_completed",
            output_dir=str(task.output_dir),
            trajectory=[],
        )

    monkeypatch.setattr(controller, "_run_autonomous_ppa", fake_ppa)

    result = _run_optimise_only(
        task,
        budget=_budget(),
        status_only=False,
        max_steps=2,
    )

    assert result.success is True
    assert result.termination_reason == "optimisation_completed"
    assert called["baseline"] == _baseline("initial")
    assert any(event.stage == "baseline_promoted" for event in result.trajectory)
