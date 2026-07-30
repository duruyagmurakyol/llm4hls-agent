#!/usr/bin/env python3

"""Advance the HLS PPA agent by one guarded candidate iteration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage(name: str, arguments: list[str], allow_failure: bool = False) -> int:
    print(f"\n=== {name} ===", flush=True)
    command = [sys.executable, *arguments]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0 and not allow_failure:
        raise SystemExit(completed.returncode)
    return completed.returncode


def candidate_indices(output_dir: Path) -> list[int]:
    pattern = re.compile(r"candidate_(\d{3})\.cpp$")
    result: list[int] = []
    for path in output_dir.glob("candidate_*.cpp"):
        match = pattern.match(path.name)
        if match:
            result.append(int(match.group(1)))
    return sorted(set(result))


def refresh_summary(config_path: Path) -> dict[str, Any]:
    run_stage(
        "Evaluate current experiment",
        [str(SCRIPTS / "evaluate_ppa_experiment.py"), str(config_path)],
    )
    config = load_json(config_path)
    summary_path = REPO_ROOT / config["output_dir"] / "experiment_summary.json"
    return load_json(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate exactly one guarded HLS PPA candidate."
    )
    parser.add_argument("config", type=Path, help="PPA optimisation JSON config")
    parser.add_argument(
        "--allow-synthesis",
        action="store_true",
        help="Permit one synthesis call after static validation and CSim pass",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = refresh_summary(config_path)
    indices = candidate_indices(output_dir)
    if not indices:
        raise RuntimeError("No previous candidate exists; initialise the experiment first.")

    previous_index = indices[-1]
    next_index = previous_index + 1
    max_candidates = int(config["budget"]["max_candidates"])
    used = int(summary["budget"]["synthesis_calls_used"])
    maximum = int(summary["budget"]["max_synthesis_calls"])

    if next_index > max_candidates:
        print(f"\nSTOP: candidate budget exhausted ({max_candidates}).")
        return

    print("\nAutonomous iteration plan")
    print(f"Previous candidate: {previous_index:03d}")
    print(f"Next candidate: {next_index:03d}")
    print(f"Synthesis budget: {used}/{maximum} used")

    run_stage(
        "Prepare evidence-driven refinement prompt",
        [
            str(SCRIPTS / "prepare_ppa_refinement.py"),
            str(config_path),
            "--previous-index",
            str(previous_index),
            "--next-index",
            str(next_index),
        ],
    )
    run_stage(
        "Generate candidate",
        [
            str(SCRIPTS / "generate_ppa_candidate.py"),
            str(config_path),
            "--candidate-index",
            str(next_index),
        ],
    )

    static_rc = run_stage(
        "Static validation",
        [
            str(SCRIPTS / "validate_ppa_candidate.py"),
            str(config_path),
            "--candidate-index",
            str(next_index),
        ],
        allow_failure=True,
    )
    if static_rc != 0:
        print("\nSTOP: candidate rejected by the static gate. No CSim or synthesis was run.")
        refresh_summary(config_path)
        return

    csim_rc = run_stage(
        "Vitis CSim",
        [
            str(SCRIPTS / "run_ppa_candidate_csim.py"),
            str(config_path),
            "--candidate-index",
            str(next_index),
        ],
        allow_failure=True,
    )
    if csim_rc != 0:
        print("\nSTOP: candidate rejected by CSim. No synthesis was run.")
        refresh_summary(config_path)
        return

    if not args.allow_synthesis:
        print("\nSTOP: candidate passed all pre-synthesis gates.")
        print("Rerun with --allow-synthesis to spend one synthesis call.")
        refresh_summary(config_path)
        return

    summary = refresh_summary(config_path)
    used = int(summary["budget"]["synthesis_calls_used"])
    maximum = int(summary["budget"]["max_synthesis_calls"])
    if used >= maximum:
        print(f"\nSTOP: synthesis budget exhausted ({used}/{maximum}).")
        return

    run_stage(
        "Vitis synthesis",
        [
            str(SCRIPTS / "run_ppa_candidate_synthesis.py"),
            str(config_path),
            "--candidate-index",
            str(next_index),
        ],
    )
    summary = refresh_summary(config_path)

    record = next(
        (
            item
            for item in summary.get("candidates", [])
            if item.get("candidate_index") == next_index
        ),
        None,
    )
    print("\n=== Iteration complete ===")
    if record:
        print(f"Candidate {next_index:03d}: {record.get('verdict')}")
        print(record.get("reason"))
    budget = summary.get("budget", {})
    print(
        "Synthesis budget: "
        f"{budget.get('synthesis_calls_used')}/{budget.get('max_synthesis_calls')} used"
    )
    print(f"Summary: {(output_dir / 'experiment_summary.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
