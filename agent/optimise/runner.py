"""Autonomous, budgeted HLS PPA optimisation loop."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _load_json(path: Path) -> dict[str, Any]:
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


def _initialise(config_path: Path, output_dir: Path) -> None:
    baseline = ensure_baseline_synthesis(config_path)
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
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_ppa_optimisation.py"), str(config_path)],
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode != 0 or not all(path.is_file() for path in required):
        raise RuntimeError("Baseline diagnosis and initial prompt generation failed.")


def _prepare_next_prompt(
    config_path: Path,
    previous: dict[str, Any],
    previous_index: int,
    next_index: int,
) -> None:
    if previous.get("verdict") in {"keep_pareto_candidate", "accept_dominates_baseline"}:
        prepare_tradeoff_prompt(config_path, previous_index, next_index)
    else:
        prepare_refinement_prompt(config_path, previous_index, next_index)


def _evaluate_candidate(
    config_path: Path,
    index: int,
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    static = validate_ppa_candidate(config_path, index)
    trajectory.append({"candidate": index, "stage": "static_validation", "passed": static["passed"]})
    if not static["passed"]:
        return evaluate_experiment(config_path)

    duplicate = check_candidate_duplicate(config_path, index)
    trajectory.append({"candidate": index, "stage": "duplicate_check", "passed": duplicate["passed"]})
    if not duplicate["passed"]:
        return evaluate_experiment(config_path)

    csim = run_candidate_csim(config_path, index)
    trajectory.append({"candidate": index, "stage": "csim", "passed": csim["passed"]})
    if not csim["passed"]:
        return evaluate_experiment(config_path)

    synthesis = run_candidate_synthesis(config_path, index)
    trajectory.append({
        "candidate": index,
        "stage": "synthesis",
        "passed": synthesis["passed"],
        "timed_out": synthesis.get("timed_out", False),
    })
    return evaluate_experiment(config_path)


def run_optimisation(
    config_path: Path,
    *,
    status_only: bool = False,
    max_steps: int | None = None,
) -> OptimisationRunResult:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory: list[dict[str, Any]] = []

    if status_only:
        summary = evaluate_experiment(config_path)
        return OptimisationRunResult(True, "status_only", "status_requested", summary, trajectory)

    _initialise(config_path, output_dir)
    summary = evaluate_experiment(config_path)
    maximum_candidates = int(config["budget"]["max_candidates"])
    step_limit = max_steps if max_steps is not None else maximum_candidates

    for _ in range(step_limit):
        summary = evaluate_experiment(config_path)
        budget = summary["budget"]
        if budget["synthesis_calls_remaining"] <= 0:
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
                    summary = _evaluate_candidate(config_path, latest, trajectory)
                    continue
                if _load_json(static_path).get("passed") is not True:
                    summary = evaluate_experiment(config_path)
                    continue
                if not csim_path.is_file() or _load_json(csim_path).get("passed") is not True:
                    summary = _evaluate_candidate(config_path, latest, trajectory)
                    continue
                if not synthesis_path.is_file():
                    synthesis = run_candidate_synthesis(config_path, latest)
                    trajectory.append({"candidate": latest, "stage": "synthesis", "passed": synthesis["passed"], "timed_out": synthesis.get("timed_out", False)})
                    summary = evaluate_experiment(config_path)
                    continue
            index = latest + 1
            if index > maximum_candidates:
                return OptimisationRunResult(True, "terminated_iteration_limit", "candidate_budget_exhausted", summary, trajectory)
            completed = [item for item in summary.get("candidates", []) if item.get("verdict") != "incomplete"]
            if not completed:
                return OptimisationRunResult(False, "failed", "no_completed_candidate_for_feedback", summary, trajectory)
            previous = completed[-1]
            _prepare_next_prompt(config_path, previous, int(previous["candidate_index"]), index)

        model_calls = len(list(output_dir.glob("candidate_*_model_metadata.json")))
        if model_calls >= maximum_candidates:
            return OptimisationRunResult(True, "terminated_budget", "model_call_budget_exhausted", summary, trajectory)
        generate_candidate(config_path, index)
        trajectory.append({"candidate": index, "stage": "generation", "passed": True})
        summary = _evaluate_candidate(config_path, index, trajectory)

        record = _record(summary, index)
        if record and record.get("verdict") == "accept_dominates_baseline":
            return OptimisationRunResult(True, "success", "candidate_dominates_baseline", summary, trajectory)

    return OptimisationRunResult(True, "terminated_step_limit", "max_agent_steps_reached", summary, trajectory)
