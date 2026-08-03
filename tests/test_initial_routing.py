from __future__ import annotations

from pathlib import Path

import pytest

from agent.budget import BudgetState
from agent.config import TaskManifest, load_task
from agent.controller import _detect_initial_condition, _record_phase_transitions
from agent.state import AgentPhase, AgentResult, TrajectoryEvent


def _task(tmp_path: Path) -> TaskManifest:
    candidate = tmp_path / "candidate.cpp"
    candidate.write_text("void kernel() {}\n", encoding="utf-8")
    return TaskManifest(
        path=tmp_path / "task.json",
        data={
            "task_id": "automatic_route_test",
            "task_kind": "functional_failure",
            "artifacts": {"source": str(candidate)},
            "adapter": {"kind": "auto"},
            "output_dir": str(tmp_path / "output"),
        },
    )


def _budget() -> BudgetState:
    return BudgetState(
        max_iterations=1,
        max_model_calls=1,
        max_csim_calls=1,
        max_cosim_calls=1,
        max_synthesis_calls=1,
        max_total_tokens=100,
    )


def _tool_result(kind: str, passed: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "passed": passed,
        "timed_out": False,
        "return_code": 0 if passed else 1,
        "failure_class": "none" if passed else f"{kind}_failed",
        "evidence": [] if passed else [f"{kind} failure"],
        "command": [kind],
        "duration_seconds": 0.1,
        "log_path": f"{kind}.log",
        "candidate_hash": "a" * 64,
    }
    if kind == "synthesis":
        result["metrics"] = {"latency_best_cycles": 4}
    if kind == "cosim":
        result["reports"] = ["cosim.rpt"] if passed else []
    return result


@pytest.mark.parametrize(
    ("failing_tool", "expected_stages", "expected_usage"),
    [
        ("csim", ["initial_csim"], (1, 0, 0)),
        ("synthesis", ["initial_csim", "initial_synthesis"], (1, 1, 0)),
        (
            "cosim",
            ["initial_csim", "initial_synthesis", "initial_cosim"],
            (1, 1, 1),
        ),
    ],
)
def test_initial_failure_routes_to_repair(
    tmp_path: Path,
    monkeypatch,
    failing_tool: str,
    expected_stages: list[str],
    expected_usage: tuple[int, int, int],
) -> None:
    calls: list[str] = []

    def fake_csim(task: TaskManifest, candidate: Path) -> dict[str, object]:
        calls.append("csim")
        return _tool_result("csim", failing_tool != "csim")

    def fake_synthesis(task: TaskManifest, candidate: Path) -> dict[str, object]:
        calls.append("synthesis")
        return _tool_result("synthesis", failing_tool != "synthesis")

    def fake_cosim(task: TaskManifest, candidate: Path) -> dict[str, object]:
        calls.append("cosim")
        return _tool_result("cosim", failing_tool != "cosim")

    monkeypatch.setattr("agent.controller.run_csim", fake_csim)
    monkeypatch.setattr("agent.controller.run_synthesis", fake_synthesis)
    monkeypatch.setattr("agent.controller.run_cosim", fake_cosim)

    budget = _budget()
    route, trajectory = _detect_initial_condition(_task(tmp_path), budget)

    assert route == "repair"
    assert calls == [stage.removeprefix("initial_") for stage in expected_stages]
    assert [event.stage for event in trajectory] == expected_stages
    assert trajectory[-1].details["route"] == "repair"
    assert trajectory[-1].details["decision_reason"] == f"{failing_tool}_failed"
    assert (
        budget.csim_calls_used,
        budget.synthesis_calls_used,
        budget.cosim_calls_used,
    ) == expected_usage


def test_all_initial_validation_passes_routes_to_optimisation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agent.controller.run_csim",
        lambda task, candidate: _tool_result("csim", True),
    )
    monkeypatch.setattr(
        "agent.controller.run_synthesis",
        lambda task, candidate: _tool_result("synthesis", True),
    )
    monkeypatch.setattr(
        "agent.controller.run_cosim",
        lambda task, candidate: _tool_result("cosim", True),
    )

    budget = _budget()
    task = _task(tmp_path)
    route, trajectory = _detect_initial_condition(task, budget)

    assert task.data["task_kind"] == "functional_failure"
    assert route == "optimise"
    assert [event.stage for event in trajectory] == [
        "initial_csim",
        "initial_synthesis",
        "initial_cosim",
        "initial_baseline",
    ]
    assert trajectory[-1].details["route"] == "optimise"
    assert trajectory[-1].details["decision_reason"] == "all_initial_validation_passed"
    assert budget.csim_calls_used == 1
    assert budget.synthesis_calls_used == 1
    assert budget.cosim_calls_used == 1

    result = AgentResult(
        task_id=task.task_id,
        success=True,
        status="success",
        termination_reason="completed",
        output_dir=str(task.output_dir),
        trajectory=[
            *trajectory,
            TrajectoryEvent(5, "generation", "passed", {"candidate": 1}),
            TrajectoryEvent(6, "synthesis", "passed", {"candidate": 1}),
        ],
    )
    result = _record_phase_transitions(task, result)
    phases = [transition.to_phase for transition in result.phase_transitions]
    assert AgentPhase.VALIDATE_INITIAL in phases
    assert AgentPhase.ESTABLISH_BASELINE in phases
    assert AgentPhase.DIAGNOSE_PPA in phases
    assert AgentPhase.GENERATE_OPTIMISATION in phases


def test_public_auto_manifests_load_without_route_selection() -> None:
    for path in (
        Path("configs/tasks/vector_add_auto_broken.json"),
        Path("configs/tasks/vector_add_auto_correct.json"),
    ):
        task = load_task(path)
        assert task.adapter_kind == "auto"
        assert set(task.data["adapter"]) == {"kind"}
        assert "repair" in task.data
        assert "optimisation" in task.data
        assert task.data["budgets"]["max_csim_calls"] >= 1
        assert task.data["budgets"]["max_synthesis_calls"] >= 1
        assert task.data["budgets"]["max_cosim_calls"] >= 1
