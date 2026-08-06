#!/usr/bin/env python3

"""Unified entry layer for autonomous repair and PPA optimisation tasks."""

from __future__ import annotations

import json
from pathlib import Path

from agent.baseline import promote_verified_baseline
from agent.budget import BudgetExceeded, BudgetState
from agent.config import TaskManifest, load_task
from agent.optimise.config_source import ppa_config_from_task
from agent.optimise.runner import run_optimisation
from agent.repair.runner import run_repair
from agent.state import AgentPhase, AgentResult, PhaseTransition, TrajectoryEvent
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


def _cosim_required(
    task: TaskManifest,
    budget: BudgetState | None = None,
) -> bool:
    """Use the package contract for Track-A and preserve legacy behaviour."""

    track_a = task.data.get("track_a")
    if isinstance(track_a, dict):
        return bool(track_a.get("requires_cosim", False))
    if budget is not None:
        return budget.max_cosim_calls > 0
    budgets = task.data.get("budgets")
    if isinstance(budgets, dict):
        return int(budgets.get("max_cosim_calls", 0)) > 0
    return True


def _write_resolved_task(task: TaskManifest) -> Path:
    snapshot = {
        "task_id": task.task_id,
        "task_root": task.data.get("task_root"),
        "artifacts": task.data["artifacts"],
        "interface": task.data["interface"],
        "target": task.data["target"],
        "model": task.data["model"],
        "budgets": task.data["budgets"],
        "track_a": task.data.get("track_a"),
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


def _phase_for_stage(stage: str) -> AgentPhase | None:
    if stage in {"initial_csim", "initial_synthesis", "initial_cosim"}:
        return AgentPhase.VALIDATE_INITIAL
    if stage in {"initial_baseline", "baseline_promoted"}:
        return AgentPhase.ESTABLISH_BASELINE
    if stage == "repair":
        return AgentPhase.REPAIR
    if stage in {"post_repair_synthesis", "post_repair_cosim"}:
        return AgentPhase.ESTABLISH_BASELINE
    if stage == "generation":
        return AgentPhase.GENERATE_OPTIMISATION
    if stage in {"static_validation", "duplicate_check", "csim", "synthesis"}:
        return AgentPhase.VALIDATE_CANDIDATE
    return None


def _phase_event_details(event: TrajectoryEvent) -> dict[str, object]:
    details: dict[str, object] = {
        "trajectory_stage": event.stage,
        "trajectory_status": event.status,
    }
    for key in (
        "candidate",
        "candidate_hash",
        "failure_class",
        "return_code",
        "timed_out",
        "route",
        "origin",
        "source",
        "project_dir",
    ):
        if key in event.details:
            details[key] = event.details[key]
    return details


def _record_phase_transitions(
    task: TaskManifest,
    result: AgentResult,
) -> AgentResult:
    """Derive explicit phase transitions without changing workflow behaviour."""
    transitions: list[PhaseTransition] = []
    current: AgentPhase | None = None

    def transition(
        target: AgentPhase,
        reason: str,
        details: dict[str, object] | None = None,
    ) -> None:
        nonlocal current
        if current == target:
            return
        transitions.append(
            PhaseTransition(
                step=len(transitions) + 1,
                from_phase=current,
                to_phase=target,
                reason=reason,
                details=details or {},
            )
        )
        current = target

    transition(
        AgentPhase.DISCOVER,
        "task_loaded",
        {
            "adapter_kind": task.adapter_kind,
            "task_kind": task.data.get("task_kind"),
        },
    )

    if task.adapter_kind == "direct_api_repair":
        transition(
            AgentPhase.DIAGNOSE,
            "task_manifest_selected_repair",
            {"task_kind": task.data.get("task_kind")},
        )
    elif task.adapter_kind in {"autonomous_ppa", "legacy_ppa"}:
        transition(
            AgentPhase.ESTABLISH_BASELINE,
            "ppa_workflow_requires_verified_baseline",
            {"adapter_kind": task.adapter_kind},
        )
    elif task.adapter_kind == "auto":
        transition(
            AgentPhase.VALIDATE_INITIAL,
            "automatic_workflow_requires_initial_validation",
            {"adapter_kind": task.adapter_kind},
        )

    for event in result.trajectory:
        details = _phase_event_details(event)
        target = _phase_for_stage(event.stage)

        if event.stage == "repair" and current == AgentPhase.VALIDATE_INITIAL:
            transition(
                AgentPhase.DIAGNOSE,
                "initial_validation_failed",
                details,
            )

        if event.stage == "generation" and current in {
            AgentPhase.VALIDATE_INITIAL,
            AgentPhase.ESTABLISH_BASELINE,
        }:
            transition(
                AgentPhase.DIAGNOSE_PPA,
                "verified_baseline_ready_for_ppa_diagnosis",
                details,
            )

        if target is not None:
            transition(
                target,
                f"trajectory_stage_{event.stage}_{event.status}",
                details,
            )

    if current == AgentPhase.VALIDATE_CANDIDATE:
        transition(
            AgentPhase.SELECT_BEST,
            "candidate_validation_completed",
            {
                "result_status": result.status,
                "success": result.success,
            },
        )

    transition(
        AgentPhase.TERMINATE,
        result.termination_reason,
        {
            "result_status": result.status,
            "success": result.success,
        },
    )
    result.current_phase = AgentPhase.TERMINATE
    result.phase_transitions = transitions
    return result


def _run_autonomous_ppa(
    task: TaskManifest,
    *,
    status_only: bool,
    max_steps: int | None,
    budget: BudgetState,
    baseline: dict[str, object] | None = None,
) -> AgentResult:
    optimisation_input: TaskManifest | Path | dict[str, object]
    if task.adapter_kind == "legacy_ppa":
        optimisation_input = _resolve(task.data["adapter"]["config"])
    elif baseline is not None:
        config = ppa_config_from_task(task)
        config["baseline"].update(
            {
                "source": baseline["source"],
                "project_dir": baseline["project_dir"],
                "metrics": baseline["metrics"],
                "candidate_hash": baseline["candidate_hash"],
                "origin": baseline["origin"],
                "verification": baseline["validation"],
            }
        )
        optimisation_input = config
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
    interface = task.data.get("interface")
    return {
        "repair_mode": "direct_api",
        "experiment_id": task.task_id,
        "benchmark_source": repair["benchmark_source"],
        "editable_files": repair["editable_files"],
        "protected_files": repair["protected_files"],
        "context_files": repair.get("context_files", repair["protected_files"]),
        "repair_constraints": repair.get("repair_constraints", []),
        "top_function": (
            interface.get("top_function") if isinstance(interface, dict) else None
        ),
        "host_validation": repair["host_validation"],
        "independent_validation": repair["independent_validation"],
        "model": model["name"],
        "temperature": model.get("temperature", 0.0),
        "max_output_tokens": model.get("max_tokens", 2048),
        "api_timeout_seconds": model.get("timeout_seconds", 120),
        "thinking_budget": model.get("thinking_budget"),
    }


def _tool_event(stage: str, result: dict[str, object]) -> TrajectoryEvent:
    details = {
        key: result[key]
        for key in (
            "command",
            "return_code",
            "timed_out",
            "failure_class",
            "evidence",
            "duration_seconds",
            "log_path",
            "candidate_hash",
            "candidate_file",
            "project_dir",
            "top_function",
            "top_csynth_xml",
            "metrics",
            "reports",
        )
        if key in result
    }
    return TrajectoryEvent(
        step=0,
        stage=stage,
        status="passed" if result["passed"] else "failed",
        details=details,
    )


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
    cosim_required = _cosim_required(task, budget)

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
        synthesis_event = _tool_event(synthesis_stage, synthesis)
        synthesis_event.step = 2
        trajectory.append(synthesis_event)

        if synthesis["passed"] and cosim_required:
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
            cosim_event = _tool_event(cosim_stage, cosim)
            cosim_event.step = 3
            trajectory.append(cosim_event)
        elif synthesis["passed"]:
            print("Post-repair co-simulation: not required by this task")

    synthesis_passed = synthesis is not None and synthesis["passed"] is True
    cosim_passed = cosim is not None and cosim["passed"] is True
    success = passed and synthesis_passed and (cosim_passed if cosim_required else True)
    status = (
        "fully_verified"
        if success
        else "cosim_failed"
        if synthesis_passed and cosim_required
        else "synthesis_failed"
        if passed
        else "repair_failed"
    )
    termination_reason = (
        "repair_synthesis_and_required_cosim_completed"
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


def _detect_initial_condition(
    task: TaskManifest,
    budget: BudgetState,
) -> tuple[str, list[TrajectoryEvent], dict[str, object] | None]:
    """Validate the submitted source and return either repair or optimise."""
    candidate = _resolve(task.data["artifacts"]["source"])
    trajectory: list[TrajectoryEvent] = []

    budget.charge_csim(stage="initial_csim")
    try:
        csim = run_csim(task, candidate)
    except Exception:
        budget.update_last_event(success=False)
        raise
    budget.update_last_event(
        success=csim["passed"] is True,
        timed_out=bool(csim["timed_out"]),
        details={"candidate_hash": csim["candidate_hash"], "log_path": csim["log_path"]},
    )
    trajectory.append(_tool_event("initial_csim", csim))
    if not csim["passed"]:
        trajectory[-1].details.update({"route": "repair", "decision_reason": "csim_failed"})
        return "repair", trajectory, None

    budget.charge_synthesis(stage="initial_synthesis")
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
    trajectory.append(_tool_event("initial_synthesis", synthesis))
    if not synthesis["passed"]:
        trajectory[-1].details.update({"route": "repair", "decision_reason": "synthesis_failed"})
        return "repair", trajectory, None

    cosim: dict[str, object] | None = None
    if _cosim_required(task, budget):
        budget.charge_cosim(stage="initial_cosim")
        try:
            cosim = run_cosim(task, candidate)
        except Exception:
            budget.update_last_event(success=False)
            raise
        budget.update_last_event(
            success=cosim["passed"] is True,
            timed_out=bool(cosim["timed_out"]),
            details={"candidate_hash": cosim["candidate_hash"], "log_path": cosim["log_path"]},
        )
        trajectory.append(_tool_event("initial_cosim", cosim))
        if not cosim["passed"]:
            trajectory[-1].details.update({"route": "repair", "decision_reason": "cosim_failed"})
            return "repair", trajectory, None

    trajectory.append(
        TrajectoryEvent(
            step=0,
            stage="initial_baseline",
            status="passed",
            details={
                "route": "optimise",
                "decision_reason": "all_required_initial_validation_passed",
                "candidate_hash": synthesis["candidate_hash"],
                "candidate_file": synthesis.get("candidate_file", str(candidate)),
                "project_dir": synthesis.get("project_dir"),
                "metrics": synthesis["metrics"],
                "cosim_required": _cosim_required(task, budget),
            },
        )
    )
    verification = {
        "source": candidate,
        "csim": csim,
        "synthesis": synthesis,
        "cosim": cosim,
    }
    return "optimise", trajectory, verification


def _repair_baseline(task: TaskManifest, result: AgentResult) -> dict[str, object]:
    repair = next(event for event in result.trajectory if event.stage == "repair")
    synthesis_event = next(
        event for event in result.trajectory if event.stage == "post_repair_synthesis"
    )
    cosim_event = next(
        (event for event in result.trajectory if event.stage == "post_repair_cosim"),
        None,
    )
    synthesis = {"passed": synthesis_event.status == "passed", **synthesis_event.details}
    cosim = (
        {"passed": cosim_event.status == "passed", **cosim_event.details}
        if cosim_event is not None
        else None
    )
    return promote_verified_baseline(
        task,
        _resolve(str(synthesis_event.details["candidate_file"])),
        origin="repaired",
        csim_passed=bool(repair.details.get("independent_validation_passed")),
        synthesis=synthesis,
        cosim=cosim,
        cosim_required=(
            bool(task.data["track_a"].get("requires_cosim", False))
            if isinstance(task.data.get("track_a"), dict)
            else cosim_event is not None
        ),
    )


def _initial_baseline(
    task: TaskManifest,
    verification: dict[str, object],
) -> dict[str, object]:
    csim = verification["csim"]
    synthesis = verification["synthesis"]
    cosim = verification.get("cosim")
    if not isinstance(csim, dict) or not isinstance(synthesis, dict):
        raise TypeError("Initial verification evidence must contain CSim and synthesis results")
    if cosim is not None and not isinstance(cosim, dict):
        raise TypeError("Initial co-simulation evidence must be structured when present")
    return promote_verified_baseline(
        task,
        Path(verification["source"]),
        origin="initial",
        csim_passed=bool(csim["passed"]),
        synthesis=synthesis,
        cosim=cosim,
        cosim_required=(
            bool(task.data["track_a"].get("requires_cosim", False))
            if isinstance(task.data.get("track_a"), dict)
            else cosim is not None
        ),
    )


def _baseline_event(baseline: dict[str, object]) -> TrajectoryEvent:
    return TrajectoryEvent(
        step=0,
        stage="baseline_promoted",
        status="passed",
        details={
            "origin": baseline["origin"],
            "source": baseline["source"],
            "candidate_hash": baseline["candidate_hash"],
            "project_dir": baseline["project_dir"],
            "metrics": baseline["metrics"],
            "validation": baseline["validation"],
        },
    )


def _prepend_initial_validation(
    result: AgentResult,
    initial: list[TrajectoryEvent],
) -> AgentResult:
    combined = [*initial, *result.trajectory]
    for index, event in enumerate(combined, 1):
        event.step = index
    result.trajectory = combined
    return result


def _merge_results(prefix: AgentResult, suffix: AgentResult) -> AgentResult:
    suffix.trajectory = [*prefix.trajectory, *suffix.trajectory]
    for index, event in enumerate(suffix.trajectory, 1):
        event.step = index
    return suffix


def _verified_baseline_result(
    task: TaskManifest,
    baseline: dict[str, object],
    *,
    termination_reason: str,
    error: BudgetExceeded | None = None,
) -> AgentResult:
    source = str(baseline["source"])
    trajectory: list[TrajectoryEvent] = []
    if error is not None:
        trajectory.append(
            TrajectoryEvent(
                step=1,
                stage="budget",
                status="passed",
                details={
                    "error": str(error),
                    "action": "retained_verified_baseline",
                },
            )
        )
    trajectory.append(
        TrajectoryEvent(
            step=len(trajectory) + 1,
            stage="select_best",
            status="passed",
            details={
                "selected_design": source,
                "selected_design_fully_verified": True,
                "selected_design_frequency_compliant": None,
                "selected_design_resource_compliant": True,
                "latest_candidate": None,
                "best_correct_candidate": source,
                "best_ppa_candidate": None,
                "pareto_archive": [],
            },
        )
    )
    return AgentResult(
        task_id=task.task_id,
        success=True,
        status="verified_baseline",
        termination_reason=termination_reason,
        output_dir=str(task.output_dir),
        trajectory=trajectory,
    )


def _load_verified_baseline(task: TaskManifest) -> dict[str, object] | None:
    path = _output_dir(task) / "verified_baseline.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(value, dict):
        return None
    source = value.get("source")
    validation = value.get("validation")
    if not isinstance(source, str) or not isinstance(validation, dict):
        return None
    if validation.get("csim_passed") is not True or validation.get("synthesis_passed") is not True:
        return None
    if validation.get("cosim_required", True) and validation.get("cosim_passed") is not True:
        return None
    return value


def _budget_fallback_reason(budget: BudgetState) -> str:
    if budget.stop_reason == "track_a_credit_budget_exhausted":
        return "verified_baseline_official_credit_budget_exhausted"
    return "verified_baseline_budget_exhausted"


def _ppa_budget_available(
    task: TaskManifest,
    budget: BudgetState,
    max_steps: int | None,
) -> bool:
    if max_steps == 0:
        return True
    return budget.can_generate_candidate(
        reserve_csim=1,
        reserve_synthesis=1,
        reserve_cosim=1 if _cosim_required(task, budget) else 0,
    )


def _run_auto(
    task: TaskManifest,
    *,
    status_only: bool,
    max_steps: int | None,
    budget: BudgetState,
) -> AgentResult:
    if status_only:
        raise ValueError("status-only is not supported for automatically discovered tasks")

    print("\n=== Initial validation ===", flush=True)
    route, initial, verification = _detect_initial_condition(task, budget)
    failed_stage = next((event.stage for event in initial if event.status == "failed"), None)

    if route == "repair":
        print(f"{failed_stage} failed; entering repair.", flush=True)
        repair_result = _run_direct_api_repair(task, budget)
        if not repair_result.success:
            return _prepend_initial_validation(repair_result, initial)

        baseline = _repair_baseline(task, repair_result)
        repair_result.trajectory.append(_baseline_event(baseline))
        print("Verified repaired source promoted to the active baseline.", flush=True)
        if not _ppa_budget_available(task, budget, max_steps):
            reason = (
                "verified_baseline_insufficient_official_credits_for_ppa"
                if budget.max_track_a_credits is not None
                else "verified_baseline_no_ppa_budget"
            )
            fallback = _verified_baseline_result(
                task,
                baseline,
                termination_reason=reason,
            )
            return _prepend_initial_validation(
                _merge_results(repair_result, fallback),
                initial,
            )

        print("Entering PPA optimisation with the repaired baseline.", flush=True)
        try:
            optimisation = _run_autonomous_ppa(
                task,
                status_only=False,
                max_steps=max_steps,
                budget=budget,
                baseline=baseline,
            )
        except BudgetExceeded as error:
            fallback = _verified_baseline_result(
                task,
                baseline,
                termination_reason=_budget_fallback_reason(budget),
                error=error,
            )
            return _prepend_initial_validation(
                _merge_results(repair_result, fallback),
                initial,
            )
        return _prepend_initial_validation(
            _merge_results(repair_result, optimisation),
            initial,
        )

    if verification is None:
        raise RuntimeError("Initial validation passed without verification evidence")
    baseline = _initial_baseline(task, verification)
    initial.append(_baseline_event(baseline))
    print("Verified initial source promoted to the active baseline.", flush=True)
    if not _ppa_budget_available(task, budget, max_steps):
        reason = (
            "verified_baseline_insufficient_official_credits_for_ppa"
            if budget.max_track_a_credits is not None
            else "verified_baseline_no_ppa_budget"
        )
        result = _verified_baseline_result(
            task,
            baseline,
            termination_reason=reason,
        )
        return _prepend_initial_validation(result, initial)

    validation_text = (
        "Initial CSim, synthesis and required co-simulation passed"
        if _cosim_required(task, budget)
        else "Initial CSim and synthesis passed; co-simulation is not required"
    )
    print(f"{validation_text}; entering PPA optimisation.", flush=True)
    try:
        result = _run_autonomous_ppa(
            task,
            status_only=False,
            max_steps=max_steps,
            budget=budget,
            baseline=baseline,
        )
    except BudgetExceeded as error:
        result = _verified_baseline_result(
            task,
            baseline,
            termination_reason=_budget_fallback_reason(budget),
            error=error,
        )
    return _prepend_initial_validation(result, initial)


def _budget_exhausted_result(
    task: TaskManifest,
    error: BudgetExceeded,
    budget: BudgetState,
) -> AgentResult:
    baseline = _load_verified_baseline(task)
    if baseline is not None:
        return _verified_baseline_result(
            task,
            baseline,
            termination_reason=_budget_fallback_reason(budget),
            error=error,
        )

    official = budget.stop_reason == "track_a_credit_budget_exhausted"
    return AgentResult(
        task_id=task.task_id,
        success=False,
        status=(
            "official_credit_budget_exhausted" if official else "budget_exhausted"
        ),
        termination_reason=(
            "official_credit_budget_exhausted" if official else "budget_exhausted"
        ),
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
        result = _budget_exhausted_result(task, error, budget)
    except Exception:
        budget.set_stop_reason("execution_error")
        budget_path = _write_budget_summary(task, budget)
        print(f"Budget summary: {budget_path.relative_to(REPO_ROOT)}")
        raise

    result = _record_phase_transitions(task, result)
    budget.set_stop_reason(result.termination_reason)
    budget_path = _write_budget_summary(task, budget)
    result_path = _write_result(result)
    print(f"Budget summary: {budget_path.relative_to(REPO_ROOT)}")
    print(f"\nUnified result: {result_path.relative_to(REPO_ROOT)}")
    return result
