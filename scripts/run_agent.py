#!/usr/bin/env python3

"""Run one unified LLM4HLS repair or optimisation task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.controller import run_agent  # noqa: E402
from agent.onboarding_safe import onboard_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the unified budgeted LLM4HLS agent."
    )
    parser.add_argument(
        "task",
        type=Path,
        help="Unified task manifest or benchmark directory containing an HLS TCL flow",
    )
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--max-agent-steps", type=int, default=None)
    parser.add_argument(
        "--onboard-only",
        action="store_true",
        help="Discover a benchmark directory and generate configuration without running the agent",
    )
    args = parser.parse_args()

    try:
        target = args.task.resolve()
        if target.is_dir():
            task_path = onboard_benchmark(target)
        else:
            if args.onboard_only:
                raise ValueError("--onboard-only requires a benchmark directory.")
            task_path = target

        if args.onboard_only:
            return

        result = run_agent(
            task_path,
            status_only=args.status_only,
            max_steps=args.max_agent_steps,
        )
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"Agent configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
