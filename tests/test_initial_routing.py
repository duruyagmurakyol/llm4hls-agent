from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.baseline import promote_verified_baseline
from agent.budget import BudgetState
from agent.config import TaskManifest, load_task, validate_task
from agent.controller import (
    _detect_initial_condition,
    _record_phase_transitions,
    _run_auto,
    _run_autonomous_ppa,
)
from agent.optimise.runner import OptimisationRunResult
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
            "interface": {"top_function": "kernel"},
            "adapter": {"kind": "auto"},
            "output_dir": str(tmp_path / "output"),
        },
    )


def _budget() -> BudgetState:
    return BudgetState(
        max_iterations=2,
        max_model_calls=2,
        max_csim_calls=2,
        max_cosim_calls=2,
        max_synthesis_calls=2,
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
        "candidate_file": "candidate.cpp",
    }
    if kind == "synthesis":
        result.update(
            metrics={"latency_best_cycles": 4},
            project_dir="synthesis_project",
            top_function="kernel",
            top_csynth_xml="kernel_csynth.xml",
        )
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
    route, trajectory, verification = _detect_initial_condition(_task(tmp_path), budget)

    assert route == "repair"
    assert verification is None
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
    route, trajectory, verification = _detect_initial_condition(task, budget)

    assert task.data["task_kind"] == "functional_failure"
    assert route == "optimise"
    assert verification is not None
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


def test_promote_verified_baseline_copies_source_and_reports(tmp_path: Path) -> None:
    source = tmp_path / "candidate.cpp"
    source.write_text("void kernel() {}\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    project = tmp_path / "synthesis_project"
    report_dir = project / "solution1/syn/report"
    report_dir.mkdir(parents=True)
    (report_dir / "kernel_csynth.xml").write_text("<Report/>\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "candidate_001_prompt.txt").write_text("stale\n", encoding="utf-8")
    task = TaskManifest(
        path=tmp_path / "task.json",
        data={
            "task_id": "promote",
            "interface": {"top_function": "kernel"},
            "output_dir": str(output),
        },
    )

    baseline = promote_verified_baseline(
        task,
        source,
        origin="repaired",
        csim_passed=True,
        synthesis={
            "passed": True,
            "candidate_hash": digest,
            "project_dir": str(project),
            "top_function": "kernel",
            "metrics": {"latency_best_cycles": 4},
        },
        cosim={"passed": True, "candidate_hash": digest},
    )

    active_source = Path(str(baseline["source"]))
    project_dir = Path(str(baseline["project_dir"]))
    assert active_source.read_bytes() == source.read_bytes()
    assert (project_dir / "solution1/syn/report/kernel_csynth.xml").is_file()
    assert baseline["candidate_hash"] == digest
    assert baseline["origin"] == "repaired"
    assert not (output / "candidate_001_prompt.txt").exists()
    persisted = json.loads((output / "verified_baseline.json").read_text())
    assert persisted["metrics"]["latency_best_cycles"] == 4


def test_auto_repair_promotes_baseline_and_enters_ppa(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _task(tmp_path)
    initial = [
        TrajectoryEvent(
            0,
            "initial_csim",
            "failed",
            {"route": "repair", "decision_reason": "csim_failed"},
        )
    ]
    repair = AgentResult(
        task_id=task.task_id,
        success=True,
        status="fully_verified",
        termination_reason="repair_synthesis_and_cosim_completed",
        output_dir=str(task.output_dir),
        trajectory=[
            TrajectoryEvent(1, "repair", "passed", {}),
            TrajectoryEvent(2, "post_repair_synthesis", "passed", {}),
            TrajectoryEvent(3, "post_repair_cosim", "passed", {}),
        ],
    )
    baseline = {
        "origin": "repaired",
        "source": "active_baseline.cpp",
        "candidate_hash": "a" * 64,
        "project_dir": "verified_baseline_project",
        "metrics": {"latency_best_cycles": 4},
        "validation": {
            "csim_passed": True,
            "synthesis_passed": True,
            "cosim_passed": True,
        },
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "agent.controller._detect_initial_condition",
        lambda supplied_task, budget: ("repair", initial, None),
    )
    monkeypatch.setattr(
        "agent.controller._run_direct_api_repair",
        lambda supplied_task, budget: repair,
    )
    monkeypatch.setattr(
        "agent.controller._repair_baseline",
        lambda supplied_task, result: baseline,
    )

    def fake_ppa(
        supplied_task: TaskManifest,
        *,
        status_only: bool,
        max_steps: int | None,
        budget: BudgetState,
        baseline: dict[str, object] | None = None,
    ) -> AgentResult:
        captured["baseline"] = baseline
        return AgentResult(
            task_id=supplied_task.task_id,
            success=True,
            status="terminated_step_limit",
            termination_reason="max_agent_steps_reached",
            output_dir=str(supplied_task.output_dir),
            trajectory=[TrajectoryEvent(1, "generation", "passed", {"candidate": 1})],
        )

    monkeypatch.setattr("agent.controller._run_autonomous_ppa", fake_ppa)
    result = _run_auto(
        task,
        status_only=False,
        max_steps=1,
        budget=_budget(),
    )

    assert captured["baseline"] == baseline
    assert [event.stage for event in result.trajectory] == [
        "initial_csim",
        "repair",
        "post_repair_synthesis",
        "post_repair_cosim",
        "baseline_promoted",
        "generation",
    ]


def test_auto_correct_promotes_initial_baseline_without_resynthesis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _task(tmp_path)
    initial = [TrajectoryEvent(0, "initial_baseline", "passed", {"route": "optimise"})]
    verification = {"source": tmp_path / "candidate.cpp"}
    baseline = {
        "origin": "initial",
        "source": "active_baseline.cpp",
        "candidate_hash": "a" * 64,
        "project_dir": "verified_baseline_project",
        "metrics": {"latency_best_cycles": 4},
        "validation": {
            "csim_passed": True,
            "synthesis_passed": True,
            "cosim_passed": True,
        },
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "agent.controller._detect_initial_condition",
        lambda supplied_task, budget: ("optimise", initial, verification),
    )
    monkeypatch.setattr(
        "agent.controller._initial_baseline",
        lambda supplied_task, supplied_verification: baseline,
    )

    def fake_ppa(
        supplied_task: TaskManifest,
        *,
        status_only: bool,
        max_steps: int | None,
        budget: BudgetState,
        baseline: dict[str, object] | None = None,
    ) -> AgentResult:
        captured["baseline"] = baseline
        return AgentResult(
            task_id=supplied_task.task_id,
            success=True,
            status="terminated_step_limit",
            termination_reason="max_agent_steps_reached",
            output_dir=str(supplied_task.output_dir),
            trajectory=[],
        )

    monkeypatch.setattr("agent.controller._run_autonomous_ppa", fake_ppa)
    result = _run_auto(
        task,
        status_only=False,
        max_steps=0,
        budget=_budget(),
    )

    assert captured["baseline"] == baseline
    assert [event.stage for event in result.trajectory] == [
        "initial_baseline",
        "baseline_promoted",
    ]


def test_autonomous_ppa_receives_promoted_baseline_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.cpp"
    source.write_text("void kernel() {}\n", encoding="utf-8")
    build = tmp_path / "task.cfg"
    build.write_text("[hls]\n", encoding="utf-8")
    task = TaskManifest(
        path=tmp_path / "task.json",
        data={
            "task_id": "ppa_handoff",
            "artifacts": {
                "source": str(source),
                "build_files": [str(build)],
            },
            "interface": {
                "top_function": "kernel",
                "protected_contract": [],
            },
            "budgets": {
                "max_iterations": 1,
                "max_csim_calls": 2,
                "max_cosim_calls": 2,
                "max_synthesis_calls": 2,
                "max_model_calls": 1,
                "max_total_tokens": None,
            },
            "model": {"name": "model"},
            "optimisation": {"validation": {}, "prompt_constraints": []},
            "adapter": {"kind": "auto"},
            "output_dir": str(tmp_path / "output"),
        },
    )
    baseline = {
        "origin": "repaired",
        "source": "output/active_baseline.cpp",
        "candidate_hash": "b" * 64,
        "project_dir": "output/verified_baseline_project",
        "metrics": {"latency_best_cycles": 4},
        "validation": {
            "csim_passed": True,
            "synthesis_passed": True,
            "cosim_passed": True,
        },
    }
    captured: dict[str, object] = {}

    def fake_run_optimisation(config, **kwargs) -> OptimisationRunResult:
        captured["config"] = config
        return OptimisationRunResult(
            success=True,
            status="terminated_step_limit",
            termination_reason="max_agent_steps_reached",
            summary={},
            trajectory=[],
        )

    monkeypatch.setattr("agent.controller.run_optimisation", fake_run_optimisation)
    _run_autonomous_ppa(
        task,
        status_only=False,
        max_steps=0,
        budget=_budget(),
        baseline=baseline,
    )

    config = captured["config"]
    assert isinstance(config, dict)
    assert config["baseline"]["source"] == baseline["source"]
    assert config["baseline"]["project_dir"] == baseline["project_dir"]
    assert config["baseline"]["metrics"] == baseline["metrics"]
    assert config["baseline"]["candidate_hash"] == baseline["candidate_hash"]


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
        assert task.data["budgets"]["max_csim_calls"] >= 2
        assert task.data["budgets"]["max_synthesis_calls"] >= 2
        assert task.data["budgets"]["max_cosim_calls"] >= 2


def test_auto_manifest_reserves_detection_and_repair_verification() -> None:
    data = json.loads(
        Path("configs/tasks/vector_add_auto_broken.json").read_text(encoding="utf-8")
    )
    data["budgets"]["max_cosim_calls"] = 1

    with pytest.raises(
        ValueError,
        match="max_cosim_calls must be at least 2",
    ):
        validate_task(data)
