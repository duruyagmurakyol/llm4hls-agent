#!/usr/bin/env python3

"""Run repeated repair-only or repair-to-optimisation agent tasks.

This is the generic entry point for suite indexes that contain either
``direct_api_repair`` tasks or unified ``auto`` tasks. It reuses the established
sequential runner and its partial-summary behaviour while extending preflight
validation and reporting to the full-agent adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_overnight_repair_suite as suite

_original_validate_task_inputs = suite.validate_task_inputs
_original_make_row = suite.make_row


def validate_task_inputs(task_path: Path) -> list[str]:
    task = suite.load_json(task_path)
    adapter = task.get("adapter")
    kind = adapter.get("kind") if isinstance(adapter, dict) else None

    errors = _original_validate_task_inputs(task_path)
    adapter_error = "adapter.kind must be direct_api_repair"

    if kind == "auto":
        errors = [error for error in errors if error != adapter_error]
        if not isinstance(task.get("optimisation"), dict):
            errors.append("auto tasks must define task.optimisation")
    elif kind != "direct_api_repair":
        errors = [error for error in errors if error != adapter_error]
        errors.append("adapter.kind must be direct_api_repair or auto")

    return errors


def make_row(**kwargs: Any) -> dict[str, Any]:
    """Add complete shared-budget and PPA outcome fields to suite summaries."""
    row = _original_make_row(**kwargs)
    task = kwargs["task"]
    output_dir = suite.resolve(str(task["output_dir"]))
    unified = suite.optional_json(output_dir / "unified_agent_result.json")
    budget = suite.optional_json(output_dir / "budget_summary.json")
    experiment = suite.optional_json(output_dir / "experiment_summary.json")

    consumed = budget.get("consumed")
    if not isinstance(consumed, dict):
        consumed = {}

    row["repair_input_tokens"] = row.get("input_tokens")
    row["repair_output_tokens"] = row.get("output_tokens")
    row["repair_total_tokens"] = row.get("total_tokens")
    row["input_tokens"] = consumed.get("input_tokens", row.get("input_tokens"))
    row["output_tokens"] = consumed.get("output_tokens", row.get("output_tokens"))
    row["total_tokens"] = consumed.get("total_tokens", row.get("total_tokens"))
    row["budget_iterations_used"] = consumed.get("iterations")
    row["budget_model_calls_used"] = consumed.get("model_calls")
    row["budget_csim_calls_used"] = consumed.get("csim_calls")
    row["budget_synthesis_calls_used"] = consumed.get("synthesis_calls")
    row["budget_cosim_calls_used"] = consumed.get("cosim_calls")
    row["budget_stop_reason"] = budget.get("stop_reason")

    trajectory = unified.get("trajectory")
    if not isinstance(trajectory, list):
        trajectory = []
    optimisation_stages = {
        "generation",
        "select_refinement_parent",
        "static_validation",
        "duplicate_check",
        "csim",
        "synthesis",
        "cosim",
    }
    row["optimisation_reached"] = any(
        isinstance(event, dict) and event.get("stage") in optimisation_stages
        for event in trajectory
    )

    candidates = experiment.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    row["candidate_count"] = len(candidates)
    row["fully_verified_candidate_count"] = sum(
        isinstance(candidate, dict) and candidate.get("fully_verified") is True
        for candidate in candidates
    )

    selected = experiment.get("selected_design")
    if isinstance(selected, dict):
        row["selected_candidate"] = selected.get("candidate_index")
        row["selected_verdict"] = selected.get("verdict")
    else:
        row["selected_candidate"] = None
        row["selected_verdict"] = None

    return row


suite.validate_task_inputs = validate_task_inputs
suite.make_row = make_row


if __name__ == "__main__":
    suite.main()
