#!/usr/bin/env python3

"""Run one unified LLM4HLS repair or optimisation task."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.controller import run_agent  # noqa: E402
from agent.onboarding_safe import onboard_benchmark  # noqa: E402
from agent.resume import resume_agent  # noqa: E402
from agent.terminal_reporting import (  # noqa: E402
    build_run_summary,
    render_run_terminal,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the unified budgeted LLM4HLS agent."
    )
    parser.add_argument(
        "task",
        type=Path,
        help="Unified task manifest or benchmark directory containing an HLS build configuration",
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
        help="Discover and print a benchmark directory without running the agent",
    )
    args = parser.parse_args()

    started = time.monotonic()
    try:
        target = args.task.resolve()
        if target.is_dir():
            task_input = onboard_benchmark(target)
        else:
            if args.onboard_only:
                raise ValueError("--onboard-only requires a benchmark directory.")
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
    except (FileNotFoundError, ValueError, KeyError) as error:
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
