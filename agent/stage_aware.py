"""Minimal stage-aware generation and repair for Track-A task types.

The normal repair and optimisation paths remain unchanged.  This module adds
one bounded source-refinement loop for task types whose decisive evidence comes
from a specific HLS stage:

- ``generate``: implement a complete kernel from the public contract;
- ``synth_fix``: feed synthesis diagnostics back to the model;
- ``structural``: feed C/RTL co-simulation diagnostics back to the model.

Every candidate is validated in the natural HLS order (CSim, synthesis, then
required co-simulation).  The first failed stage becomes the feedback for the
next attempt.  A source is promoted only after all required stages pass.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from agent.baseline import promote_verified_baseline
from agent.budget import BudgetExceeded, BudgetState
from agent.config import TaskManifest
from agent.repair.generate import generate_repair
from agent.repair.output_validation import InvalidModelOutputError
from agent.state import AgentResult, TrajectoryEvent
from agent.tools.cosim import run_cosim
from agent.tools.synthesis import run_csim, run_synthesis

REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_TASK_KINDS = {"generate", "synth_fix", "structural"}


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def supports_stage_aware_task(task: TaskManifest) -> bool:
    return str(task.data.get("task_kind", "")).strip().casefold() in SUPPORTED_TASK_KINDS


def _requires_cosim(task: TaskManifest) -> bool:
    kind = str(task.data.get("task_kind", "")).strip().casefold()
    metadata = task.data.get("track_a")
    declared = bool(metadata.get("requires_cosim", False)) if isinstance(metadata, dict) else False
    return kind == "structural" or declared


def _context_files(task: TaskManifest) -> list[Path]:
    repair = task.data.get("repair")
    values = repair.get("context_files", []) if isinstance(repair, dict) else []
    result: list[Path] = []
    root = _resolve(task.data.get("task_root", task.path.parent))
    for value in values:
        path = Path(str(value))
        path = path if path.is_absolute() else root / path
        if path.is_file() and path not in result:
            result.append(path)
    return result


def _feedback_from_report(stage: str, report: dict[str, Any]) -> dict[str, Any]:
    evidence = [str(item) for item in report.get("evidence") or []]
    log_path = report.get("log_path") or report.get("log_file")
    if log_path:
        evidence.append(f"Tool log: {log_path}")
    if report.get("timed_out") is True:
        evidence.append(
            f"The {stage} stage timed out after {report.get('timeout_seconds')} seconds."
        )
    if not evidence:
        evidence.append(
            f"{stage} failed with return code {report.get('return_code')}."
        )
    return {
        "stage": stage,
        "failure_class": str(report.get("failure_class") or "unknown"),
        "return_code": report.get("return_code"),
        "timed_out": bool(report.get("timed_out", False)),
        "evidence": evidence[:20],
    }


def _generation_feedback() -> dict[str, Any]:
    return {
        "stage": "generation",
        "failure_class": "implementation_required",
        "return_code": None,
        "timed_out": False,
        "evidence": [
            "Implement the complete kernel from the public specification, header and testbench."
        ],
    }


def _prompt(
    task: TaskManifest,
    *,
    source: Path,
    feedback: dict[str, Any],
    attempt: int,
    mode: str,
) -> tuple[str, str]:
    kind = str(task.data.get("task_kind", "")).strip().casefold()
    top = str(task.data["interface"]["top_function"])
    source_text = source.read_text(encoding="utf-8")
    context_sections = []
    for path in _context_files(task):
        context_sections.append(
            f"READ-ONLY CONTEXT FILE: {path.name}\n```\n"
            f"{path.read_text(encoding='utf-8', errors='replace')}\n```"
        )

    evidence = "\n".join(f"- {item}" for item in feedback.get("evidence", []))
    action = (
        "Implement the complete editable source from the public contract."
        if mode == "generate"
        else "Repair the editable source using the exact failing-stage evidence."
    )
    system = (
        f"Task kind: {kind}. {action} "
        "Use tool evidence as the authoritative diagnosis; do not modify protected files."
    )
    user = (
        f"TASK KIND: {kind}\n"
        f"ATTEMPT: {attempt}\n"
        f"FAILED STAGE: {feedback.get('stage')}\n"
        f"FAILURE CLASS: {feedback.get('failure_class')}\n"
        f"RETURN CODE: {feedback.get('return_code')}\n"
        f"TIMED OUT: {feedback.get('timed_out')}\n"
        "TOOL EVIDENCE:\n"
        f"{evidence}\n\n"
        f"Preserve the {top} function name and signature.\n"
        "Treat the public specification, header, testbench and build files as read-only.\n"
        "Return only the complete editable source file.\n\n"
        f"EDITABLE FILE: {source.name}\n```\n{source_text}\n```\n\n"
        + "\n\n".join(context_sections)
    )
    return system, user


def _tool_event(stage: str, report: dict[str, Any], *, attempt: int | None = None) -> TrajectoryEvent:
    details = {
        key: report.get(key)
        for key in (
            "return_code",
            "timed_out",
            "timeout_seconds",
            "failure_class",
            "evidence",
            "duration_seconds",
            "elapsed_seconds",
            "log_path",
            "log_file",
            "candidate_hash",
            "candidate_file",
            "project_dir",
            "metrics",
        )
        if key in report
    }
    if attempt is not None:
        details["attempt"] = attempt
    return TrajectoryEvent(
        step=0,
        stage=stage,
        status="passed" if report.get("passed") is True else "failed",
        details=details,
    )


def _run_tool(
    *,
    budget: BudgetState,
    kind: str,
    stage: str,
    function: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if kind == "csim":
        budget.charge_csim(stage=stage)
    elif kind == "synthesis":
        budget.charge_synthesis(stage=stage)
    elif kind == "cosim":
        budget.charge_cosim(stage=stage)
    else:
        raise ValueError(f"Unsupported tool kind: {kind}")
    try:
        report = function()
    except Exception:
        budget.update_last_event(success=False)
        raise
    budget.update_last_event(
        success=report.get("passed") is True,
        timed_out=bool(report.get("timed_out", False)),
        details={
            "candidate_hash": report.get("candidate_hash"),
            "log_path": report.get("log_path") or report.get("log_file"),
        },
    )
    return report


def _validate(
    task: TaskManifest,
    candidate: Path,
    *,
    budget: BudgetState,
    trajectory: list[TrajectoryEvent],
    prefix: str,
    attempt: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    csim = _run_tool(
        budget=budget,
        kind="csim",
        stage=f"{prefix}_csim",
        function=lambda: run_csim(task, candidate),
    )
    trajectory.append(_tool_event(f"{prefix}_csim", csim, attempt=attempt))
    if csim.get("passed") is not True:
        return csim, None, None, _feedback_from_report("csim", csim)

    synthesis = _run_tool(
        budget=budget,
        kind="synthesis",
        stage=f"{prefix}_synthesis",
        function=lambda: run_synthesis(task, candidate),
    )
    trajectory.append(_tool_event(f"{prefix}_synthesis", synthesis, attempt=attempt))
    if synthesis.get("passed") is not True:
        return csim, synthesis, None, _feedback_from_report("synthesis", synthesis)

    cosim: dict[str, Any] | None = None
    if _requires_cosim(task):
        cosim = _run_tool(
            budget=budget,
            kind="cosim",
            stage=f"{prefix}_cosim",
            function=lambda: run_cosim(task, candidate),
        )
        trajectory.append(_tool_event(f"{prefix}_cosim", cosim, attempt=attempt))
        if cosim.get("passed") is not True:
            return csim, synthesis, cosim, _feedback_from_report("cosim", cosim)

    return csim, synthesis, cosim, None


def _frequency_compliant(task: TaskManifest, metrics: dict[str, Any]) -> bool | None:
    minimum = task.data.get("target", {}).get("minimum_frequency_mhz")
    frequency = metrics.get("frequency_mhz")
    if not isinstance(frequency, (int, float)) or isinstance(frequency, bool):
        period = metrics.get("clock_period_ns")
        if isinstance(period, (int, float)) and not isinstance(period, bool) and period > 0:
            frequency = 1000.0 / period
    if not isinstance(minimum, (int, float)) or isinstance(minimum, bool):
        return None
    if not isinstance(frequency, (int, float)) or isinstance(frequency, bool):
        return None
    return float(frequency) >= float(minimum)


def _write_selected_state(
    task: TaskManifest,
    baseline: dict[str, Any],
    *,
    attempt: int,
    budget: BudgetState,
) -> dict[str, Any]:
    output_dir = _resolve(task.output_dir)
    metrics = dict(baseline.get("metrics") or {})
    requires_cosim = _requires_cosim(task)
    frequency_ok = _frequency_compliant(task, metrics)
    selected = {
        "role": "selected_design",
        "candidate_index": attempt,
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
        "verdict": f"{task.data.get('task_kind')}_verified",
        "track_a_selection": {},
        "validation": {
            "static_validation": True,
            "csim": True,
            "synthesis": True,
            "cosim": True if requires_cosim else None,
        },
    }
    state = {
        "schema_version": 4,
        "selection_policy": {
            "mode": "stage_aware_verification",
            "description": (
                "Select the first source that passes CSim, synthesis and every "
                "co-simulation stage required by the task."
            ),
        },
        "selected_design_fully_verified": True,
        "selected_design_frequency_compliant": frequency_ok,
        "selected_design_resource_compliant": True,
        "original_baseline": None,
        "latest_candidate": selected,
        "best_correct_candidate": selected,
        "best_ppa_candidate": None,
        "selected_design": selected,
        "pareto_archive": [],
    }
    _write_json(output_dir / "candidate_state.json", state)
    return selected


def _write_result_and_budget(
    task: TaskManifest,
    result: AgentResult,
    budget: BudgetState,
) -> None:
    output_dir = _resolve(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "unified_agent_result.json", result.to_dict())
    budget.write_summary(output_dir / "budget_summary.json")


def _success_result(
    task: TaskManifest,
    *,
    baseline: dict[str, Any],
    attempt: int,
    trajectory: list[TrajectoryEvent],
    budget: BudgetState,
) -> AgentResult:
    selected = _write_selected_state(task, baseline, attempt=attempt, budget=budget)
    trajectory.append(
        TrajectoryEvent(
            step=0,
            stage="baseline_promoted",
            status="passed",
            details={
                "origin": baseline["origin"],
                "source": baseline["source"],
                "candidate_hash": baseline["candidate_hash"],
                "metrics": baseline["metrics"],
                "validation": baseline["validation"],
            },
        )
    )
    trajectory.append(
        TrajectoryEvent(
            step=0,
            stage="select_best",
            status="passed",
            details={
                "selection_mode": "stage_aware_verification",
                "selected_design": selected["archived_file"],
                "selected_design_fully_verified": True,
                "selected_design_frequency_compliant": selected[
                    "meets_frequency_requirement"
                ],
                "selected_design_resource_compliant": True,
                "latest_candidate": selected["archived_file"],
                "best_correct_candidate": selected["archived_file"],
                "best_ppa_candidate": None,
                "pareto_archive": [],
            },
        )
    )
    for index, event in enumerate(trajectory, 1):
        event.step = index
    kind = str(task.data.get("task_kind"))
    result = AgentResult(
        task_id=task.task_id,
        success=True,
        status="fully_verified",
        termination_reason=f"{kind}_verified_with_stage_feedback",
        output_dir=str(task.output_dir),
        trajectory=trajectory,
    )
    budget.set_stop_reason(result.termination_reason, overwrite=True)
    _write_result_and_budget(task, result, budget)
    return result


def run_stage_aware_task(task: TaskManifest) -> AgentResult:
    """Run a bounded generate/synth-fix/structural task to full verification."""

    if not supports_stage_aware_task(task):
        raise ValueError(
            "Stage-aware execution supports only generate, synth_fix and structural tasks"
        )

    kind = str(task.data.get("task_kind", "")).strip().casefold()
    output_dir = _resolve(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "resolved_task.json",
        {
            "task_id": task.task_id,
            "task_kind": kind,
            "task_root": task.data.get("task_root"),
            "artifacts": task.data.get("artifacts"),
            "interface": task.data.get("interface"),
            "target": task.data.get("target"),
            "model": task.data.get("model"),
            "budgets": task.data.get("budgets"),
            "track_a": task.data.get("track_a"),
            "output_dir": str(task.output_dir),
        },
    )

    budget = BudgetState.from_manifest(task.data["budgets"])
    source = _resolve(task.data["artifacts"]["source"])
    if not source.is_file():
        raise FileNotFoundError(f"Editable task source not found: {source}")

    trajectory: list[TrajectoryEvent] = []
    feedback = _generation_feedback()

    try:
        if kind != "generate":
            csim, synthesis, cosim, feedback = _validate(
                task,
                source,
                budget=budget,
                trajectory=trajectory,
                prefix="initial",
            )
            if feedback is None and synthesis is not None:
                baseline = promote_verified_baseline(
                    task,
                    source,
                    origin="initial",
                    csim_passed=bool(csim and csim.get("passed") is True),
                    synthesis=synthesis,
                    cosim=cosim,
                    cosim_required=_requires_cosim(task),
                )
                return _success_result(
                    task,
                    baseline=baseline,
                    attempt=0,
                    trajectory=trajectory,
                    budget=budget,
                )

        maximum = min(budget.max_iterations, budget.max_model_calls)
        current_source = source
        last_feedback = feedback

        for attempt in range(1, maximum + 1):
            if not budget.can_generate_candidate(
                reserve_csim=1,
                reserve_synthesis=1,
                reserve_cosim=1 if _requires_cosim(task) else 0,
            ):
                budget.set_stop_reason("stage_aware_verification_budget_unavailable")
                break

            attempt_dir = output_dir / "stage_aware" / f"attempt_{attempt:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            mode = "generate" if kind == "generate" and attempt == 1 else "repair"
            system_prompt, user_prompt = _prompt(
                task,
                source=current_source,
                feedback=last_feedback,
                attempt=attempt,
                mode=mode,
            )
            (attempt_dir / "system_instruction.txt").write_text(
                system_prompt + "\n",
                encoding="utf-8",
            )
            (attempt_dir / "prompt.txt").write_text(
                user_prompt + "\n",
                encoding="utf-8",
            )

            generation_stage = f"stage_aware_attempt_{attempt:03d}_generation"
            budget.charge_iteration(stage=f"stage_aware_attempt_{attempt:03d}")
            budget.charge_model_call(stage=generation_stage)
            try:
                candidate_text, response = generate_repair(
                    model=str(task.data["model"]["name"]),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=float(task.data["model"].get("temperature", 0.0)),
                    max_tokens=int(task.data["model"].get("max_tokens", 4096)),
                    timeout_seconds=int(
                        task.data["model"].get("timeout_seconds", 180)
                    ),
                    thinking_budget=task.data["model"].get("thinking_budget"),
                    mode=mode,
                )
            except InvalidModelOutputError as error:
                budget.update_last_event(success=False)
                budget.record_model_tokens(
                    input_tokens=error.input_tokens,
                    output_tokens=error.output_tokens,
                    stage=generation_stage,
                )
                (attempt_dir / "raw_response.txt").write_text(
                    error.raw_response,
                    encoding="utf-8",
                )
                _write_json(attempt_dir / "output_validation.json", error.report)
                last_feedback = {
                    "stage": "model_output_validation",
                    "failure_class": "invalid_model_output",
                    "return_code": None,
                    "timed_out": False,
                    "evidence": [str(item) for item in error.report.get("evidence", [])],
                }
                trajectory.append(
                    TrajectoryEvent(
                        step=0,
                        stage="generation",
                        status="failed",
                        details={
                            "attempt": attempt,
                            "mode": mode,
                            "failure_class": "invalid_model_output",
                            "evidence": last_feedback["evidence"],
                        },
                    )
                )
                continue
            except Exception as error:
                budget.update_last_event(success=False)
                last_feedback = {
                    "stage": "model_generation",
                    "failure_class": type(error).__name__,
                    "return_code": None,
                    "timed_out": False,
                    "evidence": [str(error)],
                }
                trajectory.append(
                    TrajectoryEvent(
                        step=0,
                        stage="generation",
                        status="failed",
                        details={
                            "attempt": attempt,
                            "mode": mode,
                            "failure_class": type(error).__name__,
                            "evidence": [str(error)],
                        },
                    )
                )
                continue

            budget.update_last_event(success=True)
            budget.record_model_tokens(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                stage=generation_stage,
            )
            (attempt_dir / "raw_response.txt").write_text(
                response.content,
                encoding="utf-8",
            )
            _write_json(attempt_dir / "api_response.json", response.raw_response)
            candidate = attempt_dir / f"candidate{source.suffix or '.cpp'}"
            candidate.write_text(candidate_text, encoding="utf-8")
            trajectory.append(
                TrajectoryEvent(
                    step=0,
                    stage="generation",
                    status="passed",
                    details={
                        "attempt": attempt,
                        "mode": mode,
                        "candidate_file": _display(candidate),
                        "candidate_hash": _sha256(candidate),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                )
            )

            csim, synthesis, cosim, failed = _validate(
                task,
                candidate,
                budget=budget,
                trajectory=trajectory,
                prefix=f"attempt_{attempt:03d}",
                attempt=attempt,
            )
            _write_json(
                attempt_dir / "validation_summary.json",
                {
                    "candidate": _display(candidate),
                    "csim": csim,
                    "synthesis": synthesis,
                    "cosim": cosim,
                    "failed_stage": failed,
                },
            )
            if failed is not None:
                last_feedback = failed
                current_source = candidate
                continue

            if synthesis is None:
                raise RuntimeError("Verified candidate is missing synthesis evidence")
            baseline = promote_verified_baseline(
                task,
                candidate,
                origin=kind,
                csim_passed=bool(csim and csim.get("passed") is True),
                synthesis=synthesis,
                cosim=cosim,
                cosim_required=_requires_cosim(task),
            )
            return _success_result(
                task,
                baseline=baseline,
                attempt=attempt,
                trajectory=trajectory,
                budget=budget,
            )

    except BudgetExceeded:
        if budget.stop_reason is None:
            budget.set_stop_reason("stage_aware_budget_exhausted")

    for index, event in enumerate(trajectory, 1):
        event.step = index
    reason = budget.stop_reason or f"{kind}_attempt_limit_reached"
    result = AgentResult(
        task_id=task.task_id,
        success=False,
        status=f"{kind}_failed",
        termination_reason=reason,
        output_dir=str(task.output_dir),
        trajectory=trajectory,
    )
    budget.set_stop_reason(reason, overwrite=True)
    _write_result_and_budget(task, result, budget)
    return result
