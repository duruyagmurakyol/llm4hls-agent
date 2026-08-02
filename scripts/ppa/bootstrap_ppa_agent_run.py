#!/usr/bin/env python3

"""Bootstrap a clean PPA run through Candidate 001 and its first measured verdict."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "ppa"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage(title: str, args: list[str], allow_failure: bool = False) -> int:
    command = [sys.executable, *args]
    print(f"\n=== {title} ===", flush=True)
    print("Command:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0 and not allow_failure:
        raise SystemExit(completed.returncode)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialise a clean PPA experiment and evaluate Candidate 001."
    )
    parser.add_argument("config", type=Path, help="PPA optimisation config")
    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        help="Stop after Candidate 001 passes CSim",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = output_dir / "candidate_001.cpp"
    summary = output_dir / "experiment_summary.json"
    if candidate.exists() or summary.exists():
        raise FileExistsError(
            f"Bootstrap requires an empty run workspace: {output_dir}\n"
            "Use a new Track A run ID rather than overwriting evidence."
        )

    run_stage(
        "Ensure isolated baseline CSim and synthesis",
        [str(SCRIPTS / "ensure_ppa_baseline_synthesis.py"), str(config_path)],
    )
    run_stage(
        "Diagnose baseline and prepare Candidate 001 prompt",
        [str(SCRIPTS / "run_ppa_optimisation.py"), str(config_path)],
    )
    run_stage(
        "Generate Candidate 001",
        [
            str(SCRIPTS / "generate_ppa_candidate.py"),
            str(config_path),
            "--candidate-index",
            "1",
        ],
    )

    static_rc = run_stage(
        "Static validation",
        [
            str(SCRIPTS / "validate_ppa_candidate.py"),
            str(config_path),
            "--candidate-index",
            "1",
        ],
        allow_failure=True,
    )
    if static_rc != 0:
        run_stage(
            "Evaluate bootstrap result",
            [str(SCRIPTS / "evaluate_ppa_experiment.py"), str(config_path)],
        )
        print("\nBootstrap stopped: Candidate 001 failed static validation.")
        return

    duplicate_rc = run_stage(
        "Duplicate gate",
        [
            str(SCRIPTS / "detect_ppa_candidate_duplicate.py"),
            str(config_path),
            "--candidate-index",
            "1",
        ],
        allow_failure=True,
    )
    if duplicate_rc != 0:
        run_stage(
            "Evaluate bootstrap result",
            [str(SCRIPTS / "evaluate_ppa_experiment.py"), str(config_path)],
        )
        print("\nBootstrap stopped: Candidate 001 duplicated an existing source.")
        return

    csim_rc = run_stage(
        "Vitis CSim",
        [
            str(SCRIPTS / "run_ppa_candidate_csim.py"),
            str(config_path),
            "--candidate-index",
            "1",
        ],
        allow_failure=True,
    )
    if csim_rc != 0 or args.skip_synthesis:
        run_stage(
            "Evaluate bootstrap result",
            [str(SCRIPTS / "evaluate_ppa_experiment.py"), str(config_path)],
        )
        outcome = "failed CSim" if csim_rc != 0 else "passed pre-synthesis gates"
        print(f"\nBootstrap stopped: Candidate 001 {outcome}.")
        return

    run_stage(
        "Vitis synthesis",
        [
            str(SCRIPTS / "run_ppa_candidate_synthesis.py"),
            str(config_path),
            "--candidate-index",
            "1",
        ],
    )
    run_stage(
        "Evaluate bootstrap result",
        [str(SCRIPTS / "evaluate_ppa_experiment.py"), str(config_path)],
    )
    print("\nBootstrap complete: Candidate 001 has a measured verdict.")


if __name__ == "__main__":
    main()
