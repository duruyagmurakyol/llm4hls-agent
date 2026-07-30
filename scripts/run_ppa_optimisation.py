#!/usr/bin/env python3

import argparse
import json
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
    }

    for description, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    project_dir = repo_root / baseline["project_dir"]
    if not project_dir.is_dir():
        print(f"Warning: baseline project does not exist yet: {project_dir}")


def print_plan(config: dict[str, Any], repo_root: Path) -> None:
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

    print("\nPlanned workflow")
    print("1. Validate baseline")
    print("2. Synthesise baseline")
    print("3. Diagnose hierarchy")
    print("4. Map target to source")
    print("5. Generate constrained optimisation prompt")
    print("6. Generate candidate")
    print("7. Run CSim and synthesis")
    print("8. Compare PPA")
    print("9. Accept, reject, or retain candidate")
    print("10. Write experiment record")

    print("\nDry run complete. No synthesis or file modification was performed.")


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
    print_plan(config, repo_root)


if __name__ == "__main__":
    main()
