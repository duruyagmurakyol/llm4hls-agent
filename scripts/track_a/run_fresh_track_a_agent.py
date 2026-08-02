#!/usr/bin/env python3

"""Create and execute one isolated Track A agent run with a single command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str], title: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    print("Command:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialise and execute a clean FPT Track A agent run."
    )
    parser.add_argument("task", type=Path, help="Reusable Track A task manifest")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--max-agent-steps", type=int, default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create the isolated workspace without executing the agent",
    )
    args = parser.parse_args()

    task_path = args.task.resolve()
    parent_task_id = __import__("json").loads(task_path.read_text(encoding="utf-8"))["task_id"]
    generated_manifest = (
        REPO_ROOT
        / "experiments"
        / "track_a_runs"
        / str(parent_task_id)
        / args.run_id
        / "config"
        / "task.json"
    )

    run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "track_a" / "prepare_fresh_track_a_run.py"),
            str(task_path),
            "--run-id",
            args.run_id,
        ],
        "Prepare isolated competition workspace",
    )

    if args.prepare_only:
        print(f"\nPrepared manifest: {generated_manifest.relative_to(REPO_ROOT)}")
        return

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "track_a" / "run_track_a_agent.py"),
        str(generated_manifest),
    ]
    if args.max_agent_steps is not None:
        command.extend(["--max-agent-steps", str(args.max_agent_steps)])
    run(command, "Execute autonomous Track A run")

    print("\n=== Fresh Track A execution finished ===")
    print(f"Manifest: {generated_manifest.relative_to(REPO_ROOT)}")
    print(
        "Ledger: "
        + str(
            (
                generated_manifest.parents[1]
                / "ledger"
                / "run_ledger.json"
            ).relative_to(REPO_ROOT)
        )
    )


if __name__ == "__main__":
    main()
