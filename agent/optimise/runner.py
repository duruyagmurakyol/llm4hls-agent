"""Autonomous, budgeted HLS PPA optimisation loop."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.budget import BudgetState
from agent.optimise.config_source import ConfigInput, ConfigSource, as_config_source
from agent.optimise.diagnose import prepare_refinement_prompt, prepare_tradeoff_prompt
from agent.optimise.duplicate import check_candidate_duplicate
from agent.optimise.evaluate import evaluate_experiment
from agent.optimise.generate import generate_candidate
from agent.tools.synthesis import ensure_baseline_synthesis, run_candidate_csim, run_candidate_synthesis
from agent.tools.validation import validate_ppa_candidate

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OptimisationRunResult:
    success: bool
    status: str
    termination_reason: str
    summary: dict[str, Any]
    trajectory: list[dict[str, Any]]


def _load_json(path: ConfigSource | Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_indices(output_dir: Path) -> list[int]:
    pattern = re.compile(r"candidate_(\d{3})\.cpp$")
    return sorted(
        int(match.group(1))
        for path in output_dir.glob("candidate_*.cpp")
        if (match := pattern.fullmatch(path.name))
    )


def _record(summary: dict[str, Any], index: int) -> dict[str, Any] | None:
    return next(
        (item for item in summary.get("candidates", []) if item.get("candidate_index") == index),
        None,
    )


def _print_stage(index: int, stage: str, passed: bool, detail: str | None = None) -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"Candidate {index:03d} {stage}: {status}{suffix}", flush=True)


def _print_verdict(summary: dict[str, Any], index: int) -> None:
    record = _record(summary, index)
    if not record:
        print(f"Candidate {index:03d} verdict: unavailable", flush=True)
        return
    print(
        f"Candidate {index:03d} verdict: {record.get('verdict')} — {record.get('reason')}",
        flush=True,
    )


def _status_summary(config: dict[str, Any], output_dir: Path) -> tuple[str, dict[str, Any]]:
    """Return status without forcing baseline initialisation or synthesis."""
    summary_path = output_dir / "experiment_summary.json"
    if summary_path.is_file():
        return "status_only", _load_json(summary_path)

    baseline = config.get("baseline", {})
    configured_metrics = baseline.get("metrics")
    diagnosis_path = output_dir / "baseline_hierarchical_diagnosis.json"
    baseline_ready = bool(
        isinstance(configured_metrics, dict) and configured_metrics
    ) or diagnosis_path.is_file()
    status = "ready" if baseline_ready else "uninitialised"
    summary = {
        "experiment_name": config.get("experiment_name"),
        "benchmark": config.get("benchmark"),
        "status": status,
        "baseline_ready": baseline_ready,
        "baseline_metrics_available": bool(
            isinstance(configured_metrics, dict) and configured_metrics
        ),
        "diagnosis_available": diagnosis_path.is_file(),
        "candidate_count": len(_candidate_indices(output_dir)),
        "budget": {
            "max_candidates": config.get("budget", {}).get("max_candidates"),
            "max_synthesis_calls": config.get("budget", {}).get("max_synthesis_calls"),
            "synthesis_calls_used": 0,
        },
        "message": (
            "Baseline is ready; no experiment summary has been generated yet."
            if baseline_ready
            else "Baseline has not been initialised. Run without --status-only to synthesise and diagnose it."
        ),
    }
    return f"status_{status}", summary


def _initialise(
    config_source: ConfigSource,
    config: dict[str, Any],
    output_dir: Path,
    budget: BudgetState | None,
) -> None:
    project_dir = REPO_ROOT / config["baseline"]["project_dir"]
    cached = project_dir.is_dir() and any(project_dir.rglob("*_csynth.xml"))
    if budget is not None and not cached:
        budget.require("csim_calls")
        budget.require("synthesis_calls")

    baseline = ensure_baseline_synthesis(config_source)
    if budget is not None and baseline.get("cached") is not True:
        budget.charge_csim(
            stage="baseline_csim",
            success=baseline.get("passed") is True,
            timed_out=bool(baseline.get("timed_out", False)),
        )
        budget.charge_synthesis(
            stage="baseline_synthesis",
            success=baseline.get("passed") is True,
            timed_out=bool(baseline.get("timed_out", False)),
        )
    if baseline.get("passed") is not True:
        raise RuntimeError("Baseline synthesis failed.")

    required = (
        output_dir / "baseline_hierarchical_diagnosis.json",
        output_dir / "baseline_source_target.json",
        output_dir / "baseline_source_cause.json",
        output_dir / "candidate_001_prompt.txt",
    )
    if all(path.is_file() for path in required):
        return

    if isinstance(config_source, Path):
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_ppa_optimisation.py"), str(config_source)],
            cwd=REPO_ROOT,
            check=False,
        )
        if completed.returncode != 0 or not all(path.is_file() for path in required):
            raise RuntimeError("Baseline diagnosis and initial prompt generation failed.")
        return

    from scripts.run_ppa_optimisation import (
        analyse_source_causes,
        diagnose_existing_baseline,
        generate_optimisation_prompt,
        map_target_to_source,
        validate_config,
    )

    validate_config(config, REPO_ROOT)
    diagnosis = diagnose_existing_baseline(config, REPO_ROOT)
    target = map_target_to_source(config, REPO_ROOT, diagnosis)
    cause = analyse_source_causes(config, REPO_ROOT, target)
    generate_optimisation_prompt(config, REPO_ROOT, diagnosis, target, cause)
    if not all(path.is_file() for path in required):
        raise RuntimeError("Baseline diagnosis and initial prompt generation failed.")


def _prepare_next_prompt(
    config_source: ConfigSource,
    previous: dict[str, Any],
    previous_index: int,
    next_index: int,
) -> None:
    if previous.get("verdict") in {"keep_pareto_candidate", "accept_dominates_baseline"}:
        prepare_tradeoff_prompt(config_source, previous_index, next_index)
    else:
        prepare_refinement_prompt(config_source, previous_index, next_index)


def _evaluate_candidate(
    config_source: ConfigSource,
    index: int,
    trajectory: list[dict[str, Any]],
    budget: BudgetState | None,
) -> dict[str, Any]:
    print(f"\n=== Evaluate candidate {index:03d} ===", flush=True)

    static = validate_ppa_candidate(config_source, index)
    trajectory.append({"candidate": index, "stage": "static_validation", "passed": static["passed"]})
    failed_checks = [name for name, passed in (static.get("checks") or {}).items() if not passed]
    _print_stage(index, "static validation", static["passed"], ", ".join(failed_checks) or None)
    if not static["passed"]:
        summary = evaluate_experiment(config_source)
        _print_verdict(summary, index)
        return summary

    duplicate = check_candidate_duplicate(config_source, index)
    trajectory.append({"candidate": index, "stage": "duplicate_check", "passed": duplicate["passed"]})
    detail = f"duplicates candidate_{duplicate['duplicate_of']:03d}" if not duplicate["passed"] else None
    _print_stage(index, "duplicate check", duplicate["passed"], detail)
    if not duplicate["passed"]:
        summary = evaluate_experiment(config_source)
        _print_verdict(summary, index)
        return summary

    if budget is not None:
        budget.require("csim_calls")
    csim = run_candidate_csim(config_source, index)
    if budget is not None:
        budget.charge_csim(
            stage=f"candidate_{index:03d}_csim",
            success=csim["passed"],
            timed_out=bool(csim.get("timed_out", False)),
        )
    trajectory.append({"candidate": index, "stage": "csim", "passed": csim["passed"]})
    _print_stage(index, "CSim", csim["passed"], f"return code {csim.get('return_code')}")
    if not csim["passed"]:
        summary = evaluate_experiment(config_source)
        _print_verdict(summary, index)
        return summary

    if budget is not None:
        budget.require("synthesis_calls")
    synthesis = run_candidate_synthesis(config_source, index)
    if budget is not None:
        budget.charge_synthesis(
            stage=f"candidate_{index:03d}_synthesis",
            success=synthesis["passed"],
            timed_out=bool(synthesis.get("timed_out", False)),
        )
    trajectory.append({
        "candidate": index,
        "stage": "synthesis",
        "passed": synthesis["passed"],
        "timed_out": synthesis.get("timed_out", False),
    })
    synth_detail = (
        f"timed out after {synthesis.get('timeout_seconds')} seconds"
        if synthesis.get("timed_out")
        else f"return code {synthesis.get('return_code')}"
    )
    _print_stage(index, "synthesis", synthesis["passed"], synth_detail)
    summary = evaluate_experiment(config_source)
    _print_verdict(summary, index)
    return summary


def run_optimisation(
    config_input: ConfigInput,
    *,
    status_only: bool = False,
    max_steps: int | None = None,
    budget: BudgetState | None = None,
) -> OptimisationRunResult:
    config_source = as_config_source(config_input)
    config = _load_json(config_source)
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory: list[dict[str, Any]] = []

    if status_only:
        status, summary = _status_summary(config, output_dir)
        return OptimisationRunResult(True, status, "status_requested", summary, trajectory)

    _initialise(config_source, config, output_dir, budget)
    summary = evaluate_experiment(config_source)
    maximum_candidates = int(config["budget"]["max_candidates"])
    step_limit = max_steps if max_steps is not None else maximum_candidates

    for _ in range(step_limit):
        summary = evaluate_experiment(config_source)
        local_budget = summary["budget"]
        if local_budget["synthesis_calls_remaining"] <= 0:
            if budget is not None:
                budget.set_stop_reason("synthesis_budget_exhausted")
            return OptimisationRunResult(True, "terminated_budget", "synthesis_budget_exhausted", summary, trajectory)

        indices = _candidate_indices(output_dir)
        if not indices:
            index = 1
        else:
            latest = indices[-1]
            latest_record = _record(summary, latest)
            if latest_record and latest_record.get("verdict") == "incomplete":
                static_path = output_dir / f"candidate_{latest:03d}_static_validation.json"
                csim_path = output_dir / f"candidate_{latest:03d}_csim_validation.json"
                synthesis_path = output_dir / f"candidate_{latest:03d}_synthesis.json"
                if not static_path.is_file():
                    summary = _evaluate_candidate(config_source, latest, trajectory, budget)
                    continue
                if _load_json(static_path).get("passed") is not True:
                    summary = evaluate_experiment(config_source)
                    _print_verdict(summary, latest)
                    continue
                if not csim_path.is_file() or _load_json(csim_path).get("passed") is not True:
                    summary = _evaluate_candidate(config_source, latest, trajectory, budget)
                    continue
                if not synthesis_path.is_file():
                    print(f"\n=== Resume candidate {latest:03d} synthesis ===", flush=True)
                    if budget is not None:
                        budget.require("synthesis_calls")
                    synthesis = run_candidate_synthesis(config_source, latest)
                    if budget is not None:
                        budget.charge_synthesis(
                            stage=f"candidate_{latest:03d}_synthesis",
                            success=synthesis["passed"],
                            timed_out=bool(synthesis.get("timed_out", False)),
                        )
                    trajectory.append({"candidate": latest, "stage": "synthesis", "passed": synthesis["passed"], "timed_out": synthesis.get("timed_out", False)})
                    _print_stage(latest, "synthesis", synthesis["passed"])
                    summary = evaluate_experiment(config_source)
                    _print_verdict(summary, latest)
                    continue
            index = latest + 1
            if index > maximum_candidates:
                if budget is not None:
                    budget.set_stop_reason("candidate_budget_exhausted")
                return OptimisationRunResult(True, "terminated_iteration_limit", "candidate_budget_exhausted", summary, trajectory)
            completed = [item for item in summary.get("candidates", []) if item.get("verdict") != "incomplete"]
            if not completed:
                return OptimisationRunResult(False, "failed", "no_completed_candidate_for_feedback", summary, trajectory)
            previous = completed[-1]
            _prepare_next_prompt(config_source, previous, int(previous["candidate_index"]), index)

        model_calls = len(list(output_dir.glob("candidate_*_model_metadata.json")))
        if model_calls >= maximum_candidates:
            if budget is not None:
                budget.set_stop_reason("model_call_budget_exhausted")
            return OptimisationRunResult(True, "terminated_budget", "model_call_budget_exhausted", summary, trajectory)

        if budget is not None:
            if not budget.can_generate_candidate(
                reserve_csim=2,
                reserve_synthesis=2,
            ):
                budget.set_stop_reason("final_validation_budget_reserved")
                return OptimisationRunResult(
                    True,
                    "terminated_budget",
                    "final_validation_budget_reserved",
                    summary,
                    trajectory,
                )
            budget.charge_iteration(stage=f"candidate_{index:03d}_iteration")

        generate_candidate(config_source, index, budget=budget)
        trajectory.append({"candidate": index, "stage": "generation", "passed": True})
        _print_stage(index, "generation", True)
        summary = _evaluate_candidate(config_source, index, trajectory, budget)

        record = _record(summary, index)
        if record and record.get("verdict") == "accept_dominates_baseline":
            if budget is not None:
                budget.set_stop_reason("candidate_dominates_baseline")
            return OptimisationRunResult(True, "success", "candidate_dominates_baseline", summary, trajectory)

    if budget is not None:
        budget.set_stop_reason("max_agent_steps_reached")
    return OptimisationRunResult(True, "terminated_step_limit", "max_agent_steps_reached", summary, trajectory)
