#!/usr/bin/env python3

"""Run a directory of experiment JSON configs and aggregate their results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a command and capture combined stdout/stderr."""

    try:
        return subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as error:
        raise SystemExit(f"Command not found: {command[0]}") from error


def latest_result(repository_root: Path, experiment_id: str) -> Path | None:
    """Return the newest result.json for one experiment."""

    experiment_root = repository_root / "results" / "experiments" / experiment_id
    if not experiment_root.is_dir():
        return None

    candidates = sorted(experiment_root.glob("*/result.json"))
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all JSON experiment configs in a directory."
    )
    parser.add_argument("config_dir", type=Path, help="Directory containing JSON configs")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run remaining experiments after a failed experiment",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parent.parent
    config_dir = args.config_dir.expanduser().resolve()
    if not config_dir.is_dir():
        raise SystemExit(f"Config directory not found: {config_dir}")

    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        raise SystemExit(f"No JSON configs found in: {config_dir}")

    suite_name = config_dir.name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_dir = repository_root / "results" / "suites" / suite_name / timestamp
    suite_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    stopped_early = False

    for index, config_path in enumerate(configs, start=1):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        experiment_id = str(config["experiment_id"])
        print(f"[{index}/{len(configs)}] Running {experiment_id}...")

        process = run_command(
            [sys.executable, "scripts/run_experiment.py", str(config_path)],
            repository_root,
        )
        output = process.stdout or ""
        (suite_dir / f"{experiment_id}.log").write_text(output, encoding="utf-8")
        print(output, end="" if output.endswith("\n") else "\n")

        result_path = latest_result(repository_root, experiment_id)
        result: dict[str, Any] = {}
        if result_path is not None:
            result = json.loads(result_path.read_text(encoding="utf-8"))

        row = {
            "experiment_id": experiment_id,
            "config": str(config_path.relative_to(repository_root)),
            "runner_return_code": process.returncode,
            "pre_host_validation_passed": result.get("pre_host_validation_passed"),
            "agent_return_code": result.get("agent_return_code"),
            "tokens_used": result.get("tokens_used"),
            "modified_files": ";".join(result.get("modified_files", [])),
            "protected_files_unchanged": result.get("protected_files_unchanged"),
            "editable_scope_respected": result.get("editable_scope_respected"),
            "post_host_validation_passed": result.get("post_host_validation_passed"),
            "independent_validation_passed": result.get(
                "independent_validation_passed"
            ),
            "repair_diff_present": result.get("repair_diff_present"),
            "result_path": str(result_path.relative_to(repository_root))
            if result_path is not None
            else None,
            "passed": process.returncode == 0,
        }
        rows.append(row)

        if process.returncode != 0 and not args.continue_on_failure:
            stopped_early = True
            break

    passed_count = sum(bool(row["passed"]) for row in rows)
    total_run = len(rows)
    total_configured = len(configs)

    summary = {
        "schema_version": 1,
        "suite": suite_name,
        "timestamp_utc": timestamp,
        "config_directory": str(config_dir.relative_to(repository_root)),
        "configured_experiments": total_configured,
        "executed_experiments": total_run,
        "passed_experiments": passed_count,
        "failed_experiments": total_run - passed_count,
        "stopped_early": stopped_early,
        "all_passed": total_run == total_configured and passed_count == total_configured,
        "experiments": rows,
    }
    (suite_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = list(rows[0].keys()) if rows else []
    with (suite_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSuite summary")
    print(f"Results: {suite_dir.relative_to(repository_root)}")
    print(f"Passed: {passed_count}/{total_configured}")
    if stopped_early:
        print("Stopped after the first failure. Use --continue-on-failure to run all configs.")

    raise SystemExit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
