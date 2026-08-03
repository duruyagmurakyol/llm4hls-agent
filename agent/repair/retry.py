"""Budget-bounded retry loop for direct HLS repair attempts."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.budget import BudgetState
from agent.tools.reports import write_json


def _can_attempt(config: dict[str, Any], budget: BudgetState | None) -> bool:
    if budget is None:
        return True
    return budget.can_generate_candidate(
        reserve_csim=1 if config["independent_validation"].get("enabled", False) else 0,
        reserve_synthesis=0,
        reserve_cosim=0,
    )


def _attempt_limit(config: dict[str, Any], budget: BudgetState | None) -> int:
    configured = config.get("max_attempts")
    if configured is not None:
        value = int(configured)
        if value <= 0:
            raise ValueError("max_attempts must be a positive integer")
        return value
    if budget is None:
        return 1
    limits = [budget.remaining("iterations"), budget.remaining("model_calls")]
    if config["independent_validation"].get("enabled", False):
        limits.append(budget.remaining("csim_calls"))
    return min(limits)


def run_repair_loop(
    config_source: dict[str, Any] | str | Path,
    *,
    keep_workspace: bool = False,
    budget: BudgetState | None = None,
) -> tuple[bool, Path, dict[str, Any]]:
    from agent.repair.runner import REPO_ROOT, _load_config, _run_repair_once, _sha256

    config = _load_config(config_source)
    maximum = _attempt_limit(config, budget)
    if maximum <= 0:
        raise RuntimeError("Repair could not start because no attempt budget was available")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = REPO_ROOT / "results" / "experiments" / str(config["experiment_id"]) / timestamp
    run_root.mkdir(parents=True)

    attempts: list[dict[str, Any]] = []
    attempt_results: list[dict[str, Any]] = []
    seed_source: Path | None = None
    feedback: dict[str, Any] | None = None
    final_dir = run_root
    passed = False

    for attempt in range(1, maximum + 1):
        if not _can_attempt(config, budget):
            if budget is not None:
                budget.set_stop_reason("repair_budget_exhausted")
            break
        attempt_dir = run_root if maximum == 1 else run_root / f"attempt_{attempt:03d}"
        passed, final_dir, result = _run_repair_once(
            config,
            run_dir=attempt_dir,
            attempt=attempt,
            seed_source=seed_source,
            feedback=feedback,
            keep_workspace=True,
            budget=budget,
        )
        attempt_results.append(result)
        feedback = result.get("feedback")
        if not passed and not isinstance(feedback, dict):
            raise RuntimeError("Failed repair attempt did not produce structured feedback")
        candidate = final_dir / "workspace" / str(config["editable_files"][0])
        attempts.append(
            {
                "attempt": attempt,
                "run_dir": str(final_dir.relative_to(REPO_ROOT)),
                "passed": passed,
                "failed_stage": None if passed else feedback.get("stage"),
                "failure_class": "none" if passed else feedback.get("failure_class"),
                "evidence": [] if passed else feedback.get("evidence", []),
                "candidate_hash": _sha256(candidate),
            }
        )
        if passed:
            break
        seed_source = candidate

    if not attempt_results:
        raise RuntimeError("Repair could not start because no attempt budget was available")
    if not passed and budget is not None and not _can_attempt(config, budget):
        budget.set_stop_reason("repair_budget_exhausted")

    final = attempt_results[-1]
    total_input = sum(int(item.get("input_tokens") or 0) for item in attempt_results)
    total_output = sum(int(item.get("output_tokens") or 0) for item in attempt_results)
    result = {
        **final,
        "schema_version": 4,
        "failure_class": attempt_results[0]["failure_class"],
        "attempt_count": len(attempts),
        "attempts": attempts,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "tokens_used": total_input + total_output,
        "termination_reason": (
            "repair_validated"
            if passed
            else "repair_budget_exhausted"
            if budget is not None and not _can_attempt(config, budget)
            else "repair_attempt_limit_reached"
        ),
    }
    write_json(run_root / "repair_attempts.json", result)
    write_json(final_dir / "result.json", result)

    if not keep_workspace:
        for item in attempts:
            shutil.rmtree(REPO_ROOT / item["run_dir"] / "workspace")
    return passed, final_dir, result
