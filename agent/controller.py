#!/usr/bin/env python3

"""Unified entry layer for autonomous repair and PPA optimisation tasks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agent.config import TaskManifest, load_task
from agent.optimise.runner import run_optimisation
from agent.state import AgentResult, TrajectoryEvent

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _run(command: list[str], *, title: str) -> subprocess.CompletedProcess[str]:
    print(f"\n=== {title} ===", flush=True)
    print("Command:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False)


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


def _run_direct_api_repair(task: TaskManifest) -> AgentResult:
    repair_config = _resolve(task.data["adapter"]["config"])
    completed = _run(
        [sys.executable, "-m", "agent.repair.runner", str(repair_config), "--keep-workspace"],
        title="Unified repair workflow",
    )
    success = completed.returncode == 0
    return AgentResult(
        task_id=task.task_id,
        success=success,
        status="correctness_established" if success else "repair_failed",
        termination_reason="repair_completed" if success else "repair_failed",
        output_dir=str(task.output_dir),
        trajectory=[
            TrajectoryEvent(
                step=1,
                stage="repair",
                status="passed" if success else "failed",
                details={"return_code": completed.returncode, "config": str(repair_config)},
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
