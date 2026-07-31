#!/usr/bin/env python3

"""Advance the HLS PPA agent by one guarded, verdict-driven iteration."""

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
    completed = subprocess.run([sys.executable, *arguments], cwd=REPO_ROOT, check=False)
    if completed.returncode != 0 and not allow_failure:
        raise SystemExit(completed.returncode)
    return completed.returncode


def candidate_indices(output_dir: Path) -> list[int]:
    pattern = re.compile(r"candidate_(\d{3})\.cpp$")
    indices: set[int] = set()
    for path in output_dir.glob("candidate_*.cpp"):
        match = pattern.match(path.name)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def refresh_summary(config_path: Path) -> dict[str, Any]:
    run_stage(
        "Evaluate current experiment",
        [str(SCRIPTS / "evaluate_ppa_experiment.py"), str(config_path)],
    )
    config = load_json(config_path)
    return load_json(REPO_ROOT / config["output_dir"] / "experiment_summary.json")


def record_for(summary: dict[str, Any], index: int) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in summary.get("candidates", [])
            if item.get("candidate_index") == index
        ),
        None,
    )


def duplicate_gate(config_path: Path, index: int) -> bool:
    rc = run_stage(
        "Duplicate detection",
        [
            str(SCRIPTS / "detect_ppa_candidate_duplicate.py"),
            str(config_path),
            "--candidate-index",
            str(index),
        ],
        allow_failure=True,
    )
    return rc == 0


def synthesize_existing(
    config_path: Path,
    output_dir: Path,
    index: int,
    summary: dict[str, Any],
) -> None:
    budget = summary["budget"]
    used = int(budget["synthesis_calls_used"])
    maximum = int(budget["max_synthesis_calls"])
    if used >= maximum:
        print(f"\nSTOP: synthesis budget exhausted ({used}/{maximum}).")
        return

    if not duplicate_gate(config_path, index):
        print(f"\nSTOP: candidate {index:03d} rejected as a duplicate. No synthesis was run.")
        refresh_summary(config_path)
        return

    run_stage(
        "Vitis synthesis",
        [
            str(SCRIPTS / "run_ppa_candidate_synthesis.py"),
            str(config_path),
            "--candidate-index",
            str(index),
        ],
    )
    final_summary = refresh_summary(config_path)
    record = record_for(final_summary, index)
    print("\n=== Iteration complete ===")
    if record:
        print(f"Candidate {index:03d}: {record.get('verdict')}")
        print(record.get("reason"))
    final_budget = final_summary.get("budget", {})
    print(
        "Synthesis budget: "
        f"{final_budget.get('synthesis_calls_used')}/"
        f"{final_budget.get('max_synthesis_calls')} used"
    )
    print(f"Summary: {(output_dir / 'experiment_summary.json').relative_to(REPO_ROOT)}")


def prepare_prompt(
    config_path: Path,
    previous_index: int,
    next_index: int,
    previous_record: dict[str, Any],
) -> None:
    verdict = previous_record.get("verdict")
    if verdict in {"keep_pareto_candidate", "accept_dominates_baseline"}:
        run_stage(
            "Prepare Pareto trade-off refinement prompt",
            [
                str(SCRIPTS / "prepare_ppa_tradeoff_refinement.py"),
                str(config_path),
                "--source-index",
                str(previous_index),
                "--next-index",
                str(next_index),
            ],
        )
        return

    run_stage(
        "Prepare evidence-driven repair prompt",
        [
            str(SCRIPTS / "prepare_ppa_refinement.py"),
            str(config_path),
            "--previous-index",
            str(previous_index),
            "--next-index",
            str(next_index),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate, validate, and optionally synthesize one autonomous HLS PPA candidate."
    )
    parser.add_argument("config", type=Path, help="PPA optimisation JSON config")
    parser.add_argument(
        "--allow-synthesis",
        action="store_true",
        help="Permit one synthesis call after all cheap gates pass",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = refresh_summary(config_path)
    indices = candidate_indices(output_dir)
    if not indices:
        raise RuntimeError("No candidate exists; initialise the baseline experiment first.")

    latest_index = indices[-1]
    latest_record = record_for(summary, latest_index)

    # Resume a candidate that already passed cheap gates instead of generating another one.
    if latest_record and latest_record.get("verdict") == "incomplete":
        static_path = output_dir / f"candidate_{latest_index:03d}_static_validation.json"
        csim_path = output_dir / f"candidate_{latest_index:03d}_csim_validation.json"
        synthesis_path = output_dir / f"candidate_{latest_index:03d}_synthesis.json"
        static_ok = static_path.is_file() and load_json(static_path).get("passed") is True
        csim_ok = csim_path.is_file() and load_json(csim_path).get("passed") is True
        synthesis_done = synthesis_path.is_file() and load_json(synthesis_path).get("passed") is True

        if static_ok and csim_ok and not synthesis_done:
            print(f"\nResuming pre-synthesis candidate {latest_index:03d}")
            if not args.allow_synthesis:
                if duplicate_gate(config_path, latest_index):
                    print("\nSTOP: candidate passed all cheap gates and is unique.")
                    print("Rerun with --allow-synthesis to spend one synthesis call.")
                else:
                    print("\nSTOP: duplicate candidate rejected before synthesis.")
                return
            synthesize_existing(config_path, output_dir, latest_index, summary)
            return

    max_candidates = int(config["budget"]["max_candidates"])
    next_index = latest_index + 1
    if next_index > max_candidates:
        print(f"\nSTOP: candidate budget exhausted ({max_candidates}).")
        return

    # Refine the latest completed candidate, not a failed/incomplete duplicate.
    completed_records = [
        item
        for item in summary.get("candidates", [])
        if item.get("verdict") != "incomplete"
    ]
    if not completed_records:
        raise RuntimeError("No completed candidate is available for feedback refinement.")
    previous_record = completed_records[-1]
    previous_index = int(previous_record["candidate_index"])

    budget = summary["budget"]
    print("\nAutonomous iteration plan")
    print(f"Feedback source candidate: {previous_index:03d} ({previous_record.get('verdict')})")
    print(f"Next candidate: {next_index:03d}")
    print(
        "Synthesis budget: "
        f"{budget.get('synthesis_calls_used')}/{budget.get('max_synthesis_calls')} used"
    )

    prepare_prompt(config_path, previous_index, next_index, previous_record)
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

    if not duplicate_gate(config_path, next_index):
        print("\nSTOP: candidate rejected as a duplicate. No CSim or synthesis was run.")
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
        print("Rerun with --allow-synthesis; the controller will resume this candidate.")
        refresh_summary(config_path)
        return

    summary = refresh_summary(config_path)
    synthesize_existing(config_path, output_dir, next_index, summary)


if __name__ == "__main__":
    main()
