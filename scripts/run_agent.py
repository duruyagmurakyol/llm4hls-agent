#!/usr/bin/env python3

"""Run one unified LLM4HLS repair or optimisation task."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.config import TaskManifest, load_task  # noqa: E402
from agent.execution_mode import run_execution_mode  # noqa: E402
from agent.final_cosim import enforce_final_cosim_policy  # noqa: E402
from agent.onboarding_safe import onboard_benchmark  # noqa: E402
from agent.resume import resume_agent  # noqa: E402
from agent.stage_aware import (  # noqa: E402
    run_stage_aware_task,
    supports_stage_aware_task,
)
from agent.state import AgentResult, TrajectoryEvent  # noqa: E402
from agent.terminal_reporting import (  # noqa: E402
    build_run_summary,
    render_run_terminal,
)
from agent.track_a import (  # noqa: E402
    SUBMISSION_CLOCK_PERIOD_NS,
    SUBMISSION_MINIMUM_FREQUENCY_MHZ,
    is_track_a_task,
    onboard_track_a_task,
)
from agent.track_a_scoring import (  # noqa: E402
    capture_original_scoring_baseline,
    write_track_a_score_estimate,
)


def _verified_baseline_fallback(
    task_input: Any,
    error: ValueError,
) -> tuple[AgentResult, Path] | None:
    """Treat an unmappable optional PPA target as a verified-baseline stop.

    Repair and every validation stage required by the task have already
    completed before this error is raised. The promoted baseline is therefore
    the safe final design; absence of a loop-level optimisation target must not
    turn that verified result into a configuration failure.
    """

    message = str(error)
    if not (
        "Could not map diagnosis target" in message
        and "to a source loop" in message
        and hasattr(task_input, "task_id")
        and hasattr(task_input, "output_dir")
    ):
        return None

    output_dir = Path(str(task_input.output_dir)).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    baseline_path = output_dir / "verified_baseline.json"
    if not baseline_path.is_file():
        return None

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    selected_design = baseline.get("source")
    if not isinstance(selected_design, str) or not selected_design:
        return None

    result = AgentResult(
        task_id=str(task_input.task_id),
        success=True,
        status="verified_baseline",
        termination_reason="verified_baseline_no_mappable_ppa_target",
        output_dir=str(task_input.output_dir),
        trajectory=[
            TrajectoryEvent(
                step=1,
                stage="ppa_skipped",
                status="passed",
                details={
                    "reason": "no_mappable_source_loop",
                    "diagnosis_error": message,
                    "verified_baseline": str(baseline_path),
                },
            ),
            TrajectoryEvent(
                step=2,
                stage="select_best",
                status="passed",
                details={
                    "selection_mode": "research_pareto",
                    "selected_design": selected_design,
                    "selected_design_fully_verified": True,
                    "selected_design_frequency_compliant": None,
                    "selected_design_resource_compliant": True,
                    "latest_candidate": None,
                    "best_correct_candidate": selected_design,
                    "best_ppa_candidate": None,
                    "pareto_archive": [],
                },
            ),
        ],
    )
    result_path = output_dir / "unified_agent_result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return result, result_path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _format_number(value: object, *, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return "unavailable"


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _selection_mode(output_dir: Path) -> str:
    summary = _load_object(output_dir / "experiment_summary.json")
    mode = summary.get("selection_mode")
    if isinstance(mode, str) and mode:
        return mode

    state = _load_object(output_dir / "candidate_state.json")
    policy = state.get("selection_policy")
    if isinstance(policy, dict):
        mode = policy.get("mode")
        if isinstance(mode, str) and mode:
            return mode

    return "research_pareto"


def _selected_state(output_dir: Path) -> dict[str, Any]:
    state = _load_object(output_dir / "candidate_state.json")
    selected = state.get("selected_design")
    selected = selected if isinstance(selected, dict) else {}

    verified = state.get("selected_design_fully_verified")
    if not isinstance(verified, bool):
        verified = selected.get("fully_verified")

    frequency = state.get("selected_design_frequency_compliant")
    if not isinstance(frequency, bool):
        frequency = selected.get("meets_frequency_requirement")

    resources = state.get("selected_design_resource_compliant")
    if not isinstance(resources, bool):
        resources = selected.get("meets_resource_limits")

    return {
        "record": selected,
        "verified": verified if isinstance(verified, bool) else None,
        "frequency_compliant": frequency if isinstance(frequency, bool) else None,
        "resource_compliant": resources if isinstance(resources, bool) else None,
    }


def _enrich_track_a_result(
    result: AgentResult,
    path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Persist selection, submission timing and reference-harness evidence."""

    output_dir = path.parent
    budget_path = output_dir / "budget_summary.json"
    budget = _load_object(budget_path)
    track_a_budget = (
        budget.get("track_a") if isinstance(budget.get("track_a"), dict) else {}
    )
    credits = {
        "budget": track_a_budget.get("credit_budget"),
        "spent": track_a_budget.get("credits_spent"),
        "remaining": track_a_budget.get("credits_remaining"),
        "costs": dict(track_a_budget.get("credit_costs") or {}),
    }
    mode = _selection_mode(output_dir)
    selection = _selected_state(output_dir)
    final_design_verified = selection["verified"]

    if (
        result.termination_reason == "final_verification_budget_unavailable"
        and final_design_verified is True
    ):
        result.success = True
        result.status = "completed_budget"
        result.termination_reason = (
            "verified_result_selected_after_budget_exhaustion"
        )
        if budget:
            budget["stop_reason"] = result.termination_reason
            budget_path.write_text(
                json.dumps(budget, indent=2) + "\n",
                encoding="utf-8",
            )

    report = dict(report)
    selected = report.get("selected_design")
    selected = dict(selected) if isinstance(selected, dict) else {}
    estimated_frequency = selected.get("estimated_frequency_mhz")
    meets_submission_frequency = (
        estimated_frequency >= SUBMISSION_MINIMUM_FREQUENCY_MHZ
        if isinstance(estimated_frequency, (int, float))
        and not isinstance(estimated_frequency, bool)
        else None
    )
    selected.update(
        {
            "submission_clock_period_ns": SUBMISSION_CLOCK_PERIOD_NS,
            "submission_minimum_frequency_mhz": SUBMISSION_MINIMUM_FREQUENCY_MHZ,
            "meets_submission_frequency": meets_submission_frequency,
            "fully_verified": final_design_verified,
        }
    )
    for legacy_key in (
        "target_clock_period_ns",
        "target_frequency_mhz",
        "meets_target_clock",
    ):
        selected.pop(legacy_key, None)

    report["selected_design"] = selected
    report["selection_mode"] = mode
    report["final_design_verified"] = final_design_verified
    report["selected_design_frequency_compliant"] = selection[
        "frequency_compliant"
    ]
    report["selected_design_resource_compliant"] = selection[
        "resource_compliant"
    ]
    report["reference_harness_score_estimate"] = report.get(
        "public_score_estimate"
    )
    report["reference_harness_maximum_score"] = report.get("maximum_score")
    report["reference_harness_credits"] = credits
    report["submission_clock_period_ns"] = SUBMISSION_CLOCK_PERIOD_NS
    report["submission_minimum_frequency_mhz"] = (
        SUBMISSION_MINIMUM_FREQUENCY_MHZ
    )
    report["meets_submission_frequency"] = meets_submission_frequency
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    payload = result.to_dict()
    payload.update(
        {
            "selection_mode": mode,
            "final_design_verified": final_design_verified,
            "selected_design_frequency_compliant": selection[
                "frequency_compliant"
            ],
            "selected_design_resource_compliant": selection[
                "resource_compliant"
            ],
            "submission_clock_period_ns": SUBMISSION_CLOCK_PERIOD_NS,
            "submission_minimum_frequency_mhz": (
                SUBMISSION_MINIMUM_FREQUENCY_MHZ
            ),
            "meets_submission_frequency": meets_submission_frequency,
            "reference_harness_score_estimate": report.get(
                "reference_harness_score_estimate"
            ),
            "reference_harness_maximum_score": report.get(
                "reference_harness_maximum_score"
            ),
            "selected_reference_latency_cycles": selected.get(
                "official_latency_cycles"
            ),
            "reference_harness_credit_budget": credits.get("budget"),
            "reference_harness_credits_spent": credits.get("spent"),
            "reference_harness_credits_remaining": credits.get("remaining"),
            "reference_harness_score_report": _display_path(path),
        }
    )
    result_path = Path(str(result.output_dir)).expanduser()
    if not result_path.is_absolute():
        result_path = REPO_ROOT / result_path
    result_path.mkdir(parents=True, exist_ok=True)
    (result_path / "unified_agent_result.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _print_track_a_score(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected_design"]
    compliance = selected.get("meets_submission_frequency")
    compliance_text = (
        "yes"
        if compliance is True
        else "no"
        if compliance is False
        else "unavailable"
    )
    verified = report.get("final_design_verified")
    verified_text = (
        "yes"
        if verified is True
        else "no"
        if verified is False
        else "unavailable"
    )
    credits = (
        report.get("reference_harness_credits")
        if isinstance(report.get("reference_harness_credits"), dict)
        else {}
    )
    print(
        f"Reference-harness score report: {_display_path(path)}",
        flush=True,
    )
    print(f"Final selection mode: {report.get('selection_mode')}", flush=True)
    print(f"Final design fully verified: {verified_text}", flush=True)
    print(
        "Reference-harness score estimate: "
        f"{_format_number(report.get('reference_harness_score_estimate'), digits=4)} / "
        f"{_format_number(report.get('reference_harness_maximum_score'), digits=4)}",
        flush=True,
    )
    print(
        "Reference-harness latency cycles: "
        f"baseline={report['original_scoring_baseline'].get('official_latency_cycles')}, "
        f"selected={selected.get('official_latency_cycles')}",
        flush=True,
    )
    print(
        "Reference-harness credits: "
        f"{credits.get('spent', 'unavailable')} / "
        f"{credits.get('budget', 'unavailable')} "
        f"(remaining={credits.get('remaining', 'unavailable')})",
        flush=True,
    )
    print(
        "Submission frequency: "
        f"estimated={_format_number(selected.get('estimated_frequency_mhz'))} MHz, "
        f"minimum={SUBMISSION_MINIMUM_FREQUENCY_MHZ:g} MHz, "
        f"compliant={compliance_text}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the unified budgeted LLM4HLS agent."
    )
    parser.add_argument(
        "task",
        type=Path,
        help=(
            "Unified task manifest, official Track-A task package, or benchmark "
            "directory containing an HLS build configuration"
        ),
    )
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--max-agent-steps", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("auto", "repair", "optimise"),
        default="auto",
        help=(
            "Execution policy: auto preserves normal routing; repair stops after "
            "a verified repair; optimise rejects an invalid baseline and runs only PPA."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an automatic task from its saved fully verified baseline, "
            "skipping initial validation and repair"
        ),
    )
    parser.add_argument(
        "--onboard-only",
        action="store_true",
        help="Discover and print a benchmark or Track-A task directory without running the agent",
    )
    args = parser.parse_args()

    started = time.monotonic()
    task_input: Any = None
    try:
        target = args.task.resolve()
        if target.is_dir():
            task_input = (
                onboard_track_a_task(target)
                if is_track_a_task(target)
                else onboard_benchmark(target)
            )
        else:
            if args.onboard_only:
                raise ValueError("--onboard-only requires a task or benchmark directory.")
            task_input = load_task(target)

        if args.onboard_only:
            return
        if args.resume and args.status_only:
            raise ValueError("--resume cannot be combined with --status-only")
        if args.resume and args.mode == "repair":
            raise ValueError("--resume cannot be combined with --mode repair")

        if isinstance(task_input, TaskManifest):
            capture_original_scoring_baseline(task_input)

        if args.resume:
            result = resume_agent(task_input, max_steps=args.max_agent_steps)
        elif (
            isinstance(task_input, TaskManifest)
            and supports_stage_aware_task(task_input)
            and args.mode in {"auto", "repair"}
        ):
            if args.status_only:
                raise ValueError(
                    "--status-only is not supported for stage-aware task types"
                )
            result = run_stage_aware_task(task_input)
        else:
            result = run_execution_mode(
                task_input,
                mode=args.mode,
                status_only=args.status_only,
                max_steps=args.max_agent_steps,
            )
    except ValueError as error:
        fallback = _verified_baseline_fallback(task_input, error)
        if fallback is None:
            print(f"Agent configuration error: {error}", file=sys.stderr)
            raise SystemExit(2) from error
        result, result_path = fallback
        print(
            "No loop-level PPA target was supported by the synthesis evidence; "
            "retaining the fully verified repaired baseline.",
            flush=True,
        )
        print(f"Unified result: {result_path.relative_to(REPO_ROOT)}", flush=True)
    except (FileNotFoundError, KeyError) as error:
        print(f"Agent configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except RuntimeError as error:
        print(f"Agent execution failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    if isinstance(task_input, TaskManifest) and not args.status_only:
        audit = enforce_final_cosim_policy(task_input, result)
        audit_path = Path(str(task_input.output_dir)).expanduser()
        if not audit_path.is_absolute():
            audit_path = REPO_ROOT / audit_path
        audit_path = audit_path / "final_cosim_audit.json"
        print(
            "Final/Pareto co-simulation audit: "
            f"{audit.get('status')} ({_display_path(audit_path)})",
            flush=True,
        )
        if audit.get("fallback_used") is True:
            print(
                "The originally selected design failed final co-simulation; "
                "a verified Pareto/baseline fallback was selected.",
                flush=True,
            )

    if isinstance(task_input, TaskManifest):
        try:
            score = write_track_a_score_estimate(task_input, result)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Track-A score reporting failed: {error}", file=sys.stderr)
        else:
            if score is not None:
                score_path, score_report = score
                score_report = _enrich_track_a_result(
                    result,
                    score_path,
                    score_report,
                )
                _print_track_a_score(score_path, score_report)

    summary = build_run_summary(
        result,
        elapsed_seconds=time.monotonic() - started,
    )
    print("\n" + render_run_terminal(summary), flush=True)
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
