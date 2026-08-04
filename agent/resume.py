"""Resume an existing automatic task from its verified promoted baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent.budget import BudgetExceeded, BudgetState
from agent.config import TaskManifest, load_task
from agent.controller import (
    _budget_exhausted_result,
    _record_phase_transitions,
    _run_autonomous_ppa,
    _write_budget_summary,
    _write_resolved_task,
    _write_result,
)
from agent.state import AgentResult, TrajectoryEvent

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_resumable_baseline(output_dir: str | Path) -> dict[str, object]:
    """Load and validate the durable verified-baseline record."""
    root = _resolve(output_dir)
    record_path = root / "verified_baseline.json"
    if not record_path.is_file():
        raise FileNotFoundError(
            f"No verified baseline is available to resume: {record_path}"
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("verified_baseline.json must contain a JSON object")

    validation = record.get("validation")
    if not isinstance(validation, dict) or not all(
        validation.get(key) is True
        for key in ("csim_passed", "synthesis_passed", "cosim_passed")
    ):
        raise ValueError("Saved baseline is not fully verified")

    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Saved baseline has no synthesis metrics")

    source = _resolve(str(record.get("source", "")))
    project_dir = _resolve(str(record.get("project_dir", "")))
    top_report = _resolve(str(record.get("top_csynth_xml", "")))
    if not source.is_file():
        raise FileNotFoundError(f"Saved baseline source is missing: {source}")
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Saved baseline project is missing: {project_dir}")
    if not top_report.is_file():
        raise FileNotFoundError(f"Saved baseline synthesis report is missing: {top_report}")

    expected_hash = str(record.get("candidate_hash", ""))
    if not expected_hash or _sha256(source) != expected_hash:
        raise ValueError("Saved baseline source hash does not match its verification record")
    return record


def resume_agent(
    task_input: Path | TaskManifest,
    *,
    max_steps: int | None = None,
) -> AgentResult:
    """Resume PPA optimisation without rerunning initial validation or repair."""
    task = task_input if isinstance(task_input, TaskManifest) else load_task(task_input)
    if task.adapter_kind != "auto":
        raise ValueError("--resume is supported only for automatic repair-and-optimise tasks")

    baseline = load_resumable_baseline(task.output_dir)
    budget = BudgetState.from_manifest(task.data["budgets"])
    resolved_path = _write_resolved_task(task)
    print(f"Resolved task: {resolved_path.relative_to(REPO_ROOT)}")
    print(
        "Resuming PPA optimisation from the fully verified promoted baseline; "
        "initial validation and repair are skipped.",
        flush=True,
    )

    try:
        result = _run_autonomous_ppa(
            task,
            status_only=False,
            max_steps=max_steps,
            budget=budget,
            baseline=baseline,
        )
    except BudgetExceeded as error:
        result = _budget_exhausted_result(task, error)
    except Exception:
        budget.set_stop_reason("execution_error")
        budget_path = _write_budget_summary(task, budget)
        print(f"Budget summary: {budget_path.relative_to(REPO_ROOT)}")
        raise

    resume_event = TrajectoryEvent(
        step=1,
        stage="baseline_promoted",
        status="passed",
        details={
            "origin": "resumed",
            "source": baseline["source"],
            "candidate_hash": baseline["candidate_hash"],
            "project_dir": baseline["project_dir"],
            "metrics": baseline["metrics"],
            "validation": baseline["validation"],
        },
    )
    result.trajectory.insert(0, resume_event)
    for index, event in enumerate(result.trajectory, 1):
        event.step = index

    result = _record_phase_transitions(task, result)
    budget.set_stop_reason(result.termination_reason)
    budget_path = _write_budget_summary(task, budget)
    result_path = _write_result(result)
    print(f"Budget summary: {budget_path.relative_to(REPO_ROOT)}")
    print(f"\nUnified result: {result_path.relative_to(REPO_ROOT)}")
    return result
