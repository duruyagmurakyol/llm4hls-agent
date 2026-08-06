"""Resume an existing automatic task from its verified promoted baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Could not read resume state from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Resume state must contain a JSON object: {path}")
    return value


def load_resumable_baseline(output_dir: str | Path) -> dict[str, object]:
    """Load and validate the durable verified-baseline record."""
    root = _resolve(output_dir)
    record_path = root / "verified_baseline.json"
    if not record_path.is_file():
        raise FileNotFoundError(
            f"No verified baseline is available to resume: {record_path}"
        )
    record = _load_json_object(record_path)

    validation = record.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Saved baseline has no structured validation record")
    cosim_required = bool(validation.get("cosim_required", True))
    verified = bool(
        validation.get("csim_passed") is True
        and validation.get("synthesis_passed") is True
        and (
            validation.get("cosim_passed") is True
            if cosim_required
            else True
        )
    )
    if not verified:
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


def _non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Saved budget field {field} must be a non-negative integer")
    return value


def _load_resumable_budget(task: TaskManifest) -> BudgetState:
    """Restore cumulative call, token and Track-A credit usage for a retry."""

    budget = BudgetState.from_manifest(task.data["budgets"])
    summary_path = _resolve(task.output_dir) / "budget_summary.json"
    summary = _load_json_object(summary_path)
    if not summary:
        return budget

    consumed = summary.get("consumed")
    if not isinstance(consumed, dict):
        raise ValueError("Saved budget summary has no consumed section")

    counters = {
        "iterations_used": ("iterations", budget.max_iterations),
        "model_calls_used": ("model_calls", budget.max_model_calls),
        "csim_calls_used": ("csim_calls", budget.max_csim_calls),
        "cosim_calls_used": ("cosim_calls", budget.max_cosim_calls),
        "synthesis_calls_used": (
            "synthesis_calls",
            budget.max_synthesis_calls,
        ),
    }
    for attribute, (field, limit) in counters.items():
        value = _non_negative_int(consumed.get(field, 0), field=f"consumed.{field}")
        if value > limit:
            raise ValueError(
                f"Saved budget consumed.{field}={value} exceeds configured limit {limit}"
            )
        setattr(budget, attribute, value)

    budget.input_tokens_used = _non_negative_int(
        consumed.get("input_tokens", 0),
        field="consumed.input_tokens",
    )
    budget.output_tokens_used = _non_negative_int(
        consumed.get("output_tokens", 0),
        field="consumed.output_tokens",
    )
    if (
        budget.max_total_tokens is not None
        and budget.total_tokens_used > budget.max_total_tokens
    ):
        raise ValueError(
            "Saved token usage exceeds the configured total-token budget"
        )

    track_a = summary.get("track_a")
    track_a = track_a if isinstance(track_a, dict) else {}
    if budget.max_track_a_credits is not None:
        saved_costs = track_a.get("credit_costs")
        if isinstance(saved_costs, dict) and saved_costs != budget.track_a_credit_costs:
            raise ValueError(
                "Saved Track-A credit costs do not match the current task configuration"
            )
        spent_value = track_a.get("credits_spent")
        if spent_value is None:
            spent = (
                budget.csim_calls_used * budget.track_a_credit_costs["csim"]
                + budget.synthesis_calls_used
                * budget.track_a_credit_costs["synthesis"]
                + budget.cosim_calls_used * budget.track_a_credit_costs["cosim"]
            )
        else:
            spent = _non_negative_int(
                spent_value,
                field="track_a.credits_spent",
            )
        if spent > budget.max_track_a_credits:
            raise ValueError(
                "Saved Track-A credit usage exceeds the configured official budget"
            )
        budget.track_a_credits_used = spent

    events = summary.get("events")
    if isinstance(events, list):
        budget.events = [dict(item) for item in events if isinstance(item, dict)]
    budget.events.append(
        {
            "resource": "resume",
            "stage": "resume",
            "previous_track_a_credits_spent": budget.track_a_credits_used,
        }
    )
    return budget


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
    budget = _load_resumable_budget(task)
    resolved_path = _write_resolved_task(task)
    print(f"Resolved task: {resolved_path.relative_to(REPO_ROOT)}")
    print(
        "Resuming PPA optimisation from the fully verified promoted baseline; "
        "initial validation and repair are skipped.",
        flush=True,
    )
    if budget.max_track_a_credits is not None:
        print(
            "Restored official credits: "
            f"spent={budget.track_a_credits_used}, "
            f"remaining={budget.track_a_credits_remaining}",
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
        result = _budget_exhausted_result(task, error, budget)
    except Exception:
        budget.set_stop_reason("execution_error", overwrite=True)
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
            "restored_track_a_credits_spent": budget.track_a_credits_used,
        },
    )
    result.trajectory.insert(0, resume_event)
    for index, event in enumerate(result.trajectory, 1):
        event.step = index

    result = _record_phase_transitions(task, result)
    budget.set_stop_reason(result.termination_reason, overwrite=True)
    budget_path = _write_budget_summary(task, budget)
    result_path = _write_result(result)
    print(f"Budget summary: {budget_path.relative_to(REPO_ROOT)}")
    print(f"\nUnified result: {result_path.relative_to(REPO_ROOT)}")
    return result
