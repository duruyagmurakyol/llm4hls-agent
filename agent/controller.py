#!/usr/bin/env python3

"""Unified entry layer for repair and PPA task adapters."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agent.config import TaskManifest, load_task
from agent.state import AgentResult, TrajectoryEvent

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _run(command: list[str], *, title: str) -> subprocess.CompletedProcess[str]:
    print(f"\n=== {title} ===", flush=True)
    print("Command:", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout or "", end="", flush=True)
    return completed


def _write_result(result: AgentResult) -> Path:
    output_dir = _resolve(result.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "unified_agent_result.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _run_legacy_ppa(task: TaskManifest, *, status_only: bool, max_steps: int | None) -> AgentResult:
    command = [sys.executable, str(REPO_ROOT / "scripts" / "run_track_a_agent.py"), str(task.path)]
    if status_only:
        command.append("--status-only")
    if max_steps is not None:
        command.extend(["--max-agent-steps", str(max_steps)])
    completed = _run(command, title="Unified PPA workflow")
    success = completed.returncode == 0
    return AgentResult(
        task_id=task.task_id,
        success=success,
        status="completed" if success else "failed",
        termination_reason="legacy_ppa_adapter_completed" if success else "legacy_ppa_adapter_failed",
        output_dir=str(task.output_dir),
        trajectory=[TrajectoryEvent(step=1, stage="ppa_adapter", status="passed" if success else "failed", details={"return_code": completed.returncode})],
    )


def _run_direct_api_repair(task: TaskManifest) -> AgentResult:
    repair_config = _resolve(task.data["adapter"]["config"])
    command = [sys.executable, "-m", "agent.repair.runner", str(repair_config), "--keep-workspace"]
    completed = _run(command, title="Unified repair workflow")
    success = completed.returncode == 0
    return AgentResult(
        task_id=task.task_id,
        success=success,
        status="correctness_established" if success else "repair_failed",
        termination_reason="repair_completed" if success else "repair_failed",
        output_dir=str(task.output_dir),
        trajectory=[TrajectoryEvent(step=1, stage="repair", status="passed" if success else "failed", details={"return_code": completed.returncode, "config": str(repair_config)})],
    )


def run_agent(task_path: Path, *, status_only: bool = False, max_steps: int | None = None) -> AgentResult:
    task = load_task(task_path)
    if task.adapter_kind == "legacy_ppa":
        result = _run_legacy_ppa(task, status_only=status_only, max_steps=max_steps)
    elif task.adapter_kind == "direct_api_repair":
        if status_only:
            raise ValueError("status-only is not supported by direct_api_repair tasks")
        result = _run_direct_api_repair(task)
    else:
        raise ValueError(f"Unsupported adapter kind: {task.adapter_kind}")
    result_path = _write_result(result)
    print(f"\nUnified result: {result_path.relative_to(REPO_ROOT)}")
    return result
