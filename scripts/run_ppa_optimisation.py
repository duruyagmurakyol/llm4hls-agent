#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_config(config: dict[str, Any], repo_root: Path) -> None:
    required_top_level = [
        "experiment_name",
        "benchmark",
        "baseline",
        "output_dir",
        "budget",
    ]

    for key in required_top_level:
        if key not in config:
            raise ValueError(f"Missing required config field: {key}")

    baseline = config["baseline"]

    for key in ["source", "tcl", "project_dir"]:
        if key not in baseline:
            raise ValueError(f"Missing baseline field: {key}")

    required_files = {
        "baseline source": repo_root / baseline["source"],
        "baseline TCL": repo_root / baseline["tcl"],
        "hierarchical analyser": repo_root / "scripts/analyse_hls_hierarchy.py",
    }

    for description, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")


def print_configuration(config: dict[str, Any], repo_root: Path) -> None:
    baseline = config["baseline"]
    budget = config["budget"]

    print("\nPPA optimisation configuration")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Benchmark: {config['benchmark']}")
    print(f"Repository: {repo_root}")

    print("\nBaseline")
    print(f"Source: {baseline['source']}")
    print(f"TCL: {baseline['tcl']}")
    print(f"Project: {baseline['project_dir']}")

    print("\nBudget")
    print(f"Maximum candidates: {budget['max_candidates']}")
    print(f"Maximum synthesis calls: {budget['max_synthesis_calls']}")


def diagnose_existing_baseline(config: dict[str, Any], repo_root: Path) -> Path:
    project_dir = repo_root / config["baseline"]["project_dir"]

    if not project_dir.is_dir():
        raise FileNotFoundError(
            "Baseline synthesis project does not exist yet: "
            f"{project_dir}\n"
            "This stage only reuses existing reports and does not run Vitis."
        )

    reports = sorted(project_dir.rglob("*csynth.xml"))
    if not reports:
        raise FileNotFoundError(
            "No synthesis reports were found under: "
            f"{project_dir}\n"
            "This stage only reuses existing reports and does not run Vitis."
        )

    output_dir = repo_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnosis_path = output_dir / "baseline_hierarchical_diagnosis.json"

    print("\nBaseline report validation")
    print(f"Synthesis reports found: {len(reports)}")
    for report in reports:
        print(f"  - {report.relative_to(repo_root)}")

    command = [
        sys.executable,
        str(repo_root / "scripts/analyse_hls_hierarchy.py"),
        str(project_dir),
        "--output",
        str(diagnosis_path),
    ]

    print("\nRunning hierarchical diagnosis...")
    subprocess.run(command, cwd=repo_root, check=True)

    if not diagnosis_path.is_file():
        raise RuntimeError(
            f"Diagnosis command finished but output was not created: {diagnosis_path}"
        )

    return diagnosis_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diagnosis-guided HLS PPA optimisation."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the optimisation JSON config.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()

    config = load_config(config_path)
    validate_config(config, repo_root)
    print_configuration(config, repo_root)

    diagnosis_path = diagnose_existing_baseline(config, repo_root)

    print("\nStage complete")
    print("Validated the existing baseline synthesis reports.")
    print("Ran the hierarchical bottleneck diagnosis.")
    print(f"Output: {diagnosis_path.relative_to(repo_root)}")
    print("No Vitis synthesis, LLM call, or candidate modification was performed.")


if __name__ == "__main__":
    main()
