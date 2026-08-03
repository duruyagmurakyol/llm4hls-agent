#!/usr/bin/env python3

"""Unified entry layer for autonomous repair and PPA optimisation tasks."""

from __future__ import annotations

import json
from pathlib import Path

from agent.config import TaskManifest, load_task
from agent.optimise.runner import run_optimisation
from agent.repair.runner import run_repair
from agent.state import AgentResult, TrajectoryEvent

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _write_result(result: AgentResult) -> Path:
    output_dir = _resolve(result.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "unified_agent_result.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _run_autonomous_ppa(
    task: TaskManifest,
    *,
    status_only: bool,
    max_steps: int | None,
) -> AgentResult:
    config_path = _resolve(task.data["adapter"]["config"])
    optimisation = run_optimisation(
        config_path,
        status_only=status_only,
        max_steps=max_steps,
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


def _run_direct_api_repair(task: TaskManifest) -> AgentResult:
    print("\n=== Unified repair workflow ===", flush=True)
    passed, run_dir, repair_result = run_repair(
        _repair_config(task),
        keep_workspace=True,
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

    return AgentResult(
        task_id=task.task_id,
        success=passed,
        status="correctness_established" if passed else "repair_failed",
        termination_reason="repair_completed" if passed else "repair_failed",
        output_dir=str(task.output_dir),
        trajectory=[
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
        ],
    )


def run_agent(
    task_path: Path,
    *,
    status_only: bool = False,
    max_steps: int | None = None,
) -> AgentResult:
    task = load_task(task_path)
    if task.adapter_kind in {"autonomous_ppa", "legacy_ppa"}:
        result = _run_autonomous_ppa(task, status_only=status_only, max_steps=max_steps)
    elif task.adapter_kind == "direct_api_repair":
        if status_only:
            raise ValueError("status-only is not supported by direct_api_repair tasks")
        result = _run_direct_api_repair(task)
    else:
        raise ValueError(f"Unsupported adapter kind: {task.adapter_kind}")
    result_path = _write_result(result)
    print(f"\nUnified result: {result_path.relative_to(REPO_ROOT)}")
    return result
