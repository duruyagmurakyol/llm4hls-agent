"""Explicit execution policies for fair repair and optimisation experiments.

The default ``auto`` policy delegates to the established controller unchanged.
``repair`` and ``optimise`` reuse the same validation, repair, baseline and PPA
functions while changing only the decision made after initial validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from agent import controller
from agent.budget import BudgetExceeded, BudgetState
from agent.config import TaskManifest, load_task
from agent.state import AgentResult, TrajectoryEvent

ExecutionMode = Literal["auto", "repair", "optimise"]
EXECUTION_MODES = {"auto", "repair", "optimise"}


def _normalise_mode(mode: str) -> ExecutionMode:
    value = mode.strip().casefold()
    if value not in EXECUTION_MODES:
        raise ValueError(
            "execution mode must be one of: " + ", ".join(sorted(EXECUTION_MODES))
        )
    return value  # type: ignore[return-value]


def _frequency_compliant(
    task: TaskManifest,
    metrics: dict[str, object],
) -> bool | None:
    minimum = task.data.get("target", {}).get("minimum_frequency_mhz")
    frequency = metrics.get("frequency_mhz")
    if not isinstance(frequency, (int, float)) or isinstance(frequency, bool):
        period = metrics.get("clock_period_ns")
        if (
            isinstance(period, (int, float))
            and not isinstance(period, bool)
            and period > 0
        ):
            frequency = 1000.0 / period
    if not isinstance(minimum, (int, float)) or isinstance(minimum, bool):
        return None
    if not isinstance(frequency, (int, float)) or isinstance(frequency, bool):
        return None
    return float(frequency) >= float(minimum)


def _write_verified_state(
    task: TaskManifest,
    baseline: dict[str, object],
    *,
    budget: BudgetState,
    mode: ExecutionMode,
) -> dict[str, object]:
    output_dir = controller._output_dir(task)
    metrics = dict(baseline.get("metrics") or {})
    frequency_ok = _frequency_compliant(task, metrics)
    validation = baseline.get("validation")
    validation = validation if isinstance(validation, dict) else {}
    selected = {
        "role": "selected_design",
        "candidate_index": 0,
        "candidate_file": baseline["source"],
        "archived_file": baseline["source"],
        "candidate_hash": baseline["candidate_hash"],
        "metrics": metrics,
        "fully_verified": True,
        "meets_frequency_requirement": frequency_ok,
        "meets_resource_limits": True,
        "resource_limit_compliance": {
            "configured": False,
            "passed": True,
            "limits": {},
            "usage": {},
            "violations": [],
        },
        "cost": {
            "input_tokens": budget.input_tokens_used,
            "output_tokens": budget.output_tokens_used,
            "total_tokens": budget.total_tokens_used,
            "tool_calls": (
                budget.csim_calls_used
                + budget.synthesis_calls_used
                + budget.cosim_calls_used
            ),
        },
        "verdict": "already_verified" if baseline.get("origin") == "initial" else "repair_verified",
        "track_a_selection": {},
        "validation": {
            "static_validation": True,
            "csim": validation.get("csim_passed"),
            "synthesis": validation.get("synthesis_passed"),
            "cosim": validation.get("cosim_passed"),
        },
    }
    state = {
        "schema_version": 4,
        "selection_policy": {
            "mode": f"{mode}_only",
            "description": (
                "Select the verified baseline and stop without entering PPA search."
            ),
        },
        "selected_design_fully_verified": True,
        "selected_design_frequency_compliant": frequency_ok,
        "selected_design_resource_compliant": True,
        "original_baseline": selected if baseline.get("origin") == "initial" else None,
        "latest_candidate": selected,
        "best_correct_candidate": selected,
        "best_ppa_candidate": None,
        "selected_design": selected,
        "pareto_archive": [],
    }
    (output_dir / "candidate_state.json").write_text(
        __import__("json").dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )
    return selected


def _verified_stop(
    task: TaskManifest,
    baseline: dict[str, object],
    *,
    budget: BudgetState,
    termination_reason: str,
) -> AgentResult:
    selected = _write_verified_state(task, baseline, budget=budget, mode="repair")
    result = controller._verified_baseline_result(
        task,
        baseline,
        termination_reason=termination_reason,
    )
    select_event = next(
        event for event in result.trajectory if event.stage == "select_best"
    )
    select_event.details.update(
        {
            "selection_mode": "repair_only",
            "selected_design": selected["archived_file"],
            "selected_design_fully_verified": True,
            "selected_design_frequency_compliant": selected[
                "meets_frequency_requirement"
            ],
            "selected_design_resource_compliant": True,
            "latest_candidate": selected["archived_file"],
            "best_correct_candidate": selected["archived_file"],
        }
    )
    return result


def _run_repair_only(
    task: TaskManifest,
    *,
    budget: BudgetState,
    status_only: bool,
) -> AgentResult:
    if status_only:
        raise ValueError("--status-only is not supported with --mode repair")

    print("\n=== Initial validation (repair-only mode) ===", flush=True)
    route, initial, verification = controller._detect_initial_condition(task, budget)
    failed_stage = next(
        (event.stage for event in initial if event.status == "failed"),
        None,
    )

    if route == "repair":
        print(f"{failed_stage} failed; entering repair-only workflow.", flush=True)
        repair_result = controller._run_direct_api_repair(task, budget)
        if not repair_result.success:
            return controller._prepend_initial_validation(repair_result, initial)

        baseline = controller._repair_baseline(task, repair_result)
        repair_result.trajectory.append(controller._baseline_event(baseline))
        stop = _verified_stop(
            task,
            baseline,
            budget=budget,
            termination_reason="repair_only_verified",
        )
        return controller._prepend_initial_validation(
            controller._merge_results(repair_result, stop),
            initial,
        )

    if verification is None:
        raise RuntimeError("Initial validation passed without verification evidence")
    baseline = controller._initial_baseline(task, verification)
    initial.append(controller._baseline_event(baseline))
    print("Initial source is already verified; no model or PPA call is needed.", flush=True)
    result = _verified_stop(
        task,
        baseline,
        budget=budget,
        termination_reason="repair_only_already_verified",
    )
    return controller._prepend_initial_validation(result, initial)


def _invalid_baseline_result(
    task: TaskManifest,
    initial: list[TrajectoryEvent],
) -> AgentResult:
    failed = next((event for event in initial if event.status == "failed"), None)
    details = {
        "execution_mode": "optimise",
        "action": "rejected_without_repair",
        "failed_stage": failed.stage if failed is not None else None,
        "failure_class": (
            failed.details.get("failure_class") if failed is not None else None
        ),
    }
    result = AgentResult(
        task_id=task.task_id,
        success=False,
        status="invalid_optimisation_baseline",
        termination_reason="optimisation_baseline_invalid",
        output_dir=str(task.output_dir),
        trajectory=[
            TrajectoryEvent(
                step=1,
                stage="mode_guard",
                status="failed",
                details=details,
            )
        ],
    )
    return controller._prepend_initial_validation(result, initial)


def _run_optimise_only(
    task: TaskManifest,
    *,
    budget: BudgetState,
    status_only: bool,
    max_steps: int | None,
) -> AgentResult:
    if status_only:
        raise ValueError("--status-only is not supported with --mode optimise")

    print("\n=== Initial validation (optimisation-only mode) ===", flush=True)
    route, initial, verification = controller._detect_initial_condition(task, budget)
    if route != "optimise":
        print(
            "Initial source is invalid; optimisation-only mode will not repair it.",
            flush=True,
        )
        return _invalid_baseline_result(task, initial)

    if verification is None:
        raise RuntimeError("Initial validation passed without verification evidence")
    baseline = controller._initial_baseline(task, verification)
    initial.append(controller._baseline_event(baseline))
    print("Verified initial source promoted; entering PPA optimisation.", flush=True)
    try:
        result = controller._run_autonomous_ppa(
            task,
            status_only=False,
            max_steps=max_steps,
            budget=budget,
            baseline=baseline,
        )
    except BudgetExceeded as error:
        result = controller._verified_baseline_result(
            task,
            baseline,
            termination_reason=controller._budget_fallback_reason(budget),
            error=error,
        )
    return controller._prepend_initial_validation(result, initial)


def run_execution_mode(
    task_input: Path | TaskManifest,
    *,
    mode: str = "auto",
    status_only: bool = False,
    max_steps: int | None = None,
) -> AgentResult:
    """Run one task under the selected experiment policy."""

    selected_mode = _normalise_mode(mode)
    if selected_mode == "auto":
        return controller.run_agent(
            task_input,
            status_only=status_only,
            max_steps=max_steps,
        )

    task = task_input if isinstance(task_input, TaskManifest) else load_task(task_input)
    if task.adapter_kind != "auto":
        compatible = (
            selected_mode == "repair" and task.adapter_kind == "direct_api_repair"
        ) or (
            selected_mode == "optimise"
            and task.adapter_kind in {"autonomous_ppa", "legacy_ppa"}
        )
        if compatible:
            return controller.run_agent(
                task,
                status_only=status_only,
                max_steps=max_steps,
            )
        raise ValueError(
            f"--mode {selected_mode} is incompatible with adapter {task.adapter_kind}"
        )

    budget = BudgetState.from_manifest(task.data["budgets"])
    resolved_path = controller._write_resolved_task(task)
    print(f"Resolved task: {resolved_path.relative_to(controller.REPO_ROOT)}")

    try:
        result = (
            _run_repair_only(
                task,
                budget=budget,
                status_only=status_only,
            )
            if selected_mode == "repair"
            else _run_optimise_only(
                task,
                budget=budget,
                status_only=status_only,
                max_steps=max_steps,
            )
        )
    except BudgetExceeded as error:
        result = controller._budget_exhausted_result(task, error, budget)
    except Exception:
        budget.set_stop_reason("execution_error")
        controller._write_budget_summary(task, budget)
        raise

    result = controller._record_phase_transitions(task, result)
    budget.set_stop_reason(result.termination_reason)
    controller._write_budget_summary(task, budget)
    controller._write_result(result)
    return result
