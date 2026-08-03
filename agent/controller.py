#!/usr/bin/env python3

"""Unified entry layer for autonomous repair and PPA optimisation tasks."""

from __future__ import annotations

import json
from pathlib import Path

from agent.budget import BudgetExceeded, BudgetState
from agent.config import TaskManifest, load_task
from agent.optimise.runner import run_optimisation
from agent.repair.runner import run_repair
from agent.state import AgentResult, TrajectoryEvent
from agent.tools.cosim import run_cosim
from agent.tools.synthesis import run_csim, run_synthesis

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _output_dir(task: TaskManifest) -> Path:
    path = _resolve(task.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_resolved_task(task: TaskManifest) -> Path:
    snapshot = {
        "task_id": task.task_id,
        "task_root": task.data.get("task_root"),
        "artifacts": task.data["artifacts"],
        "interface": task.data["interface"],
        "target": task.data["target"],
        "model": task.data["model"],
        "budgets": task.data["budgets"],
        "output_dir": str(task.output_dir),
    }
    path = _output_dir(task) / "resolved_task.json"
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return path


def _write_result(result: AgentResult) -> Path:
    output_dir = _resolve(result.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "unified_agent_result.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _write_budget_summary(task: TaskManifest, budget: BudgetState) -> Path:
    return budget.write_summary(_output_dir(task) / "budget_summary.json")


def _run_autonomous_ppa(
    task: TaskManifest,
    *,
    status_only: bool,
    max_steps: int | None,
    budget: BudgetState,
) -> AgentResult:
    optimisation_input: TaskManifest | Path
    if task.adapter_kind == "legacy_ppa":
        optimisation_input = _resolve(task.data["adapter"]["config"])
    else:
        optimisation_input = task

    optimisation = run_optimisation(
        optimisation_input,
        status_only=status_only,
        max_steps=max_steps,
        budget=budget,
    )
    trajectory = [
        TrajectoryEvent(
            step=index,
            stage=str(item.get("stage", "optimisation")),
            status="passed" if item.get("passed", True) else "failed",
            details=item,
        )
        for index, item in enumerate(optimisation.trajectory, 1)
    ]
    return AgentResult(
        task_id=task.task_id,
        success=optimisation.success,
        status=optimisation.status,
        termination_reason=optimisation.termination_reason,
        output_dir=str(task.output_dir),
        trajectory=trajectory,
    )


def _repair_config(task: TaskManifest) -> dict[str, object]:
    repair = task.data["repair"]
    model = task.data["model"]
    return {
        "repair_mode": "direct_api",
        "experiment_id": task.task_id,
        "benchmark_source": repair["benchmark_source"],
        "editable_files": repair["editable_files"],
        "protected_files": repair["protected_files"],
        "context_files": repair.get("context_files", repair["protected_files"]),
        "host_validation": repair["host_validation"],
        "independent_validation": repair["independent_validation"],
        "model": model["name"],
        "temperature": model.get("temperature", 0.0),
        "max_output_tokens": model.get("max_tokens", 2048),
        "api_timeout_seconds": model.get("timeout_seconds", 120),
        "thinking_budget": model.get("thinking_budget"),
    }


def _run_direct_api_repair(
    task: TaskManifest,
    budget: BudgetState,
) -> AgentResult:
    print("\n=== Unified repair workflow ===", flush=True)
    passed, run_dir, repair_result = run_repair(
        _repair_config(task),
        keep_workspace=True,
        budget=budget,
    )
    print(f"Experiment: {repair_result['experiment_id']}")
    print(f"Model: {repair_result['model']}")
    print(f"Failure class: {repair_result['failure_class']}")
    print(
        "Tokens: "
        f"{repair_result['tokens_used']} "
        f"(input={repair_result['input_tokens']}, output={repair_result['output_tokens']})"
    )
    print(f"Modified files: {', '.join(repair_result['modified_files']) if repair_result['modified_files'] else 'none'}")
    print(f"Post-repair host test passed: {repair_result['post_host_validation_passed']}")
    print(f"Independent validation passed: {repair_result['independent_validation_passed']}")
    print(f"Results: {run_dir.relative_to(REPO_ROOT)}")

    trajectory = [
        TrajectoryEvent(
            step=1,
            stage="repair",
            status="passed" if passed else "failed",
            details={
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "failure_class": repair_result["failure_class"],
                "tokens_used": repair_result["tokens_used"],
                "modified_files": repair_result["modified_files"],
                "post_host_validation_passed": repair_result["post_host_validation_passed"],
                "independent_validation_passed": repair_result["independent_validation_passed"],
            },
        )
    ]
    synthesis: dict[str, object] | None = None
    cosim: dict[str, object] | None = None

    if passed:
        candidate = run_dir / "workspace" / task.data["repair"]["editable_files"][0]
        synthesis_stage = "post_repair_synthesis"
        budget.charge_synthesis(stage=synthesis_stage)
        try:
            synthesis = run_synthesis(task, candidate)
        except Exception:
            budget.update_last_event(success=False)
            raise
        budget.update_last_event(
            success=synthesis["passed"] is True,
            timed_out=bool(synthesis["timed_out"]),
            details={
                "candidate_hash": synthesis["candidate_hash"],
                "log_path": synthesis["log_path"],
            },
        )
        print(f"Post-repair synthesis passed: {synthesis['passed']}")
        print(f"Synthesis metrics: {synthesis['metrics']}")
        trajectory.append(
            TrajectoryEvent(
                step=2,
                stage=synthesis_stage,
                status="passed" if synthesis["passed"] else "failed",
                details={
                    "return_code": synthesis["return_code"],
                    "timed_out": synthesis["timed_out"],
                    "failure_class": synthesis["failure_class"],
                    "evidence": synthesis["evidence"],
                    "duration_seconds": synthesis["duration_seconds"],
                    "log_path": synthesis["log_path"],
                    "candidate_hash": synthesis["candidate_hash"],
                    "metrics": synthesis["metrics"],
                },
            )
        )

        if synthesis["passed"] and budget.max_cosim_calls > 0:
            cosim_stage = "post_repair_cosim"
            budget.charge_cosim(stage=cosim_stage)
            try:
                cosim = run_cosim(task, candidate)
            except Exception:
                budget.update_last_event(success=False)
                raise
            budget.update_last_event(
                success=cosim["passed"] is True,
                timed_out=bool(cosim["timed_out"]),
                details={
                    "candidate_hash": cosim["candidate_hash"],
                    "log_path": cosim["log_path"],
                },
            )
            print(f"Post-repair co-simulation passed: {cosim['passed']}")
            trajectory.append(
                TrajectoryEvent(
                    step=3,
                    stage=cosim_stage,
                    status="passed" if cosim["passed"] else "failed",
                    details={
                        "return_code": cosim["return_code"],
                        "timed_out": cosim["timed_out"],
                        "failure_class": cosim["failure_class"],
                        "evidence": cosim["evidence"],
                        "duration_seconds": cosim["duration_seconds"],
                        "log_path": cosim["log_path"],
                        "candidate_hash": cosim["candidate_hash"],
                        "reports": cosim["reports"],
                    },
                )
            )

    synthesis_passed = synthesis is not None and synthesis["passed"] is True
    cosim_required = budget.max_cosim_calls > 0
    cosim_passed = cosim is not None and cosim["passed"] is True
    success = passed and synthesis_passed and (cosim_passed if cosim_required else True)
    status = (
        "fully_verified"
        if success and cosim_required
        else "correctness_and_synthesis_established"
        if success
        else "cosim_failed"
        if synthesis_passed and cosim_required
        else "synthesis_failed"
        if passed
        else "repair_failed"
    )
    termination_reason = (
        "repair_synthesis_and_cosim_completed"
        if success and cosim_required
        else "repair_and_synthesis_completed"
        if success
        else "post_repair_cosim_failed"
        if synthesis_passed and cosim_required
        else "post_repair_synthesis_failed"
        if passed
        else "repair_failed"
    )
    return AgentResult(
        task_id=task.task_id,
        success=success,
        status=status,
        termination_reason=termination_reason,
        output_dir=str(task.output_dir),
        trajectory=trajectory,
    )


def _initial_csim(task: TaskManifest, budget: BudgetState) -> dict[str, object]:
    budget.require("csim_calls")
    result = run_csim(task, _resolve(task.data["artifacts"]["source"]))
    budget.charge_csim(
        stage="initial_csim",
        success=result["passed"] is True,
        timed_out=bool(result["timed_out"]),
    )
    return result


def _prepend_initial_csim(
    result: AgentResult,
    csim: dict[str, object],
) -> AgentResult:
    for index, event in enumerate(result.trajectory, 2):
        event.step = index
    result.trajectory.insert(
        0,
        TrajectoryEvent(
            step=1,
            stage="initial_csim",
            status="passed" if csim["passed"] else "failed",
            details={
                "command": csim["command"],
                "return_code": csim["return_code"],
                "timed_out": csim["timed_out"],
                "failure_class": csim["failure_class"],
                "evidence": csim["evidence"],
                "duration_seconds": csim["duration_seconds"],
                "log_path": csim["log_path"],
                "candidate_hash": csim["candidate_hash"],
            },
        ),
    )
    return result


def _run_auto(
    task: TaskManifest,
    *,
    status_only: bool,
    max_steps: int | None,
    budget: BudgetState,
) -> AgentResult:
    if status_only:
        raise ValueError("status-only is not supported for automatically discovered tasks")

    print("\n=== Initial CSim ===", flush=True)
    csim = _initial_csim(task, budget)
    if not csim["passed"]:
        print("Initial CSim failed; entering repair.", flush=True)
        return _prepend_initial_csim(_run_direct_api_repair(task, budget), csim)

    print("Initial CSim passed; entering PPA optimisation.", flush=True)
    result = _run_autonomous_ppa(
        task,
        status_only=False,
        max_steps=max_steps,
        budget=budget,
    )
    return _prepend_initial_csim(result, csim)


def _budget_exhausted_result(
    task: TaskManifest,
    error: BudgetExceeded,
) -> AgentResult:
    return AgentResult(
        task_id=task.task_id,
        success=False,
        status="budget_exhausted",
        termination_reason="budget_exhausted",
        output_dir=str(task.output_dir),
        trajectory=[
            TrajectoryEvent(
                step=1,
                stage="budget",
                status="failed",
                details={"error": str(error)},
            )
        ],
    )


def run_agent(
    task_input: Path | TaskManifest,
    *,
    status_only: bool = False,
    max_steps: int | None = None,
) -> AgentResult:
    task = task_input if isinstance(task_input, TaskManifest) else load_task(task_input)
    budget = BudgetState.from_manifest(task.data["budgets"])
    resolved_path = _write_resolved_task(task)
    print(f"Resolved task: {resolved_path.relative_to(REPO_ROOT)}")

    try:
        if task.adapter_kind in {"autonomous_ppa", "legacy_ppa"}:
            result = _run_autonomous_ppa(
                task,
                status_only=status_only,
                max_steps=max_steps,
                budget=budget,
            )
        elif task.adapter_kind == "direct_api_repair":
            if status_only:
                raise ValueError("status-only is not supported by direct_api_repair tasks")
            result = _run_direct_api_repair(task, budget)
        elif task.adapter_kind == "auto":
            result = _run_auto(
                task,
                status_only=status_only,
                max_steps=max_steps,
                budget=budget,
            )
        else:
            raise ValueError(f"Unsupported adapter kind: {task.adapter_kind}")
    except BudgetExceeded as error:
        result = _budget_exhausted_result(task, error)
    except Exception:
        budget.set_stop_reason("execution_error")
        budget_path = _write_budget_summary(task, budget)
        print(f"Budget summary: {budget_path.relative_to(REPO_ROOT)}")
        raise

    budget.set_stop_reason(result.termination_reason)
    budget_path = _write_budget_summary(task, budget)
    result_path = _write_result(result)
    print(f"Budget summary: {budget_path.relative_to(REPO_ROOT)}")
    print(f"\nUnified result: {result_path.relative_to(REPO_ROOT)}")
    return result
