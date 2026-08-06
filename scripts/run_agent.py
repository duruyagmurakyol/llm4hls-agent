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

from agent.controller import run_agent  # noqa: E402
from agent.onboarding_safe import onboard_benchmark  # noqa: E402
from agent.resume import resume_agent  # noqa: E402
from agent.state import AgentResult, TrajectoryEvent  # noqa: E402
from agent.terminal_reporting import (  # noqa: E402
    build_run_summary,
    render_run_terminal,
)
from agent.track_a import (  # noqa: E402
    is_track_a_task,
    onboard_track_a_task,
)


def _verified_baseline_fallback(
    task_input: Any,
    error: ValueError,
) -> tuple[AgentResult, Path] | None:
    """Treat an unmappable optional PPA target as a verified-baseline stop.

    Repair, synthesis and co-simulation have already completed before this
    error is raised. The promoted baseline is therefore the safe final design;
    absence of a loop-level optimisation target must not turn that verified
    result into a configuration failure.
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
            task_input = target

        if args.onboard_only:
            return
        if args.resume and args.status_only:
            raise ValueError("--resume cannot be combined with --status-only")

        result = (
            resume_agent(task_input, max_steps=args.max_agent_steps)
            if args.resume
            else run_agent(
                task_input,
                status_only=args.status_only,
                max_steps=args.max_agent_steps,
            )
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

    summary = build_run_summary(
        result,
        elapsed_seconds=time.monotonic() - started,
    )
    print("\n" + render_run_terminal(summary), flush=True)
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
