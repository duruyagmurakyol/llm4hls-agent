#!/usr/bin/env python3

import argparse
import json
import re
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


def select_recommended_target(diagnosis: dict[str, Any]) -> dict[str, Any]:
    ranked = diagnosis.get("ranked_targets")
    if isinstance(ranked, list) and ranked:
        first = ranked[0]
        if isinstance(first, dict):
            return first

    recommended = diagnosis.get("recommended_focus")
    if isinstance(recommended, dict):
        return recommended

    raise ValueError(
        "Could not find a recommended target in the hierarchical diagnosis JSON."
    )


def get_target_name(target: dict[str, Any]) -> str:
    for key in ("function", "target", "name", "report_name"):
        value = target.get(key)
        if isinstance(value, str) and value:
            return value

    raise ValueError("The selected diagnosis target has no usable name field.")


def infer_loop_label(target_name: str) -> str:
    match = re.search(r"Pipeline_(.+)$", target_name)
    if match:
        return match.group(1)

    return target_name.split("_")[-1]


def find_labelled_loop(lines: list[str], label: str) -> tuple[int, int]:
    label_pattern = re.compile(rf"^\s*{re.escape(label)}\s*:\s*$")
    label_index = next(
        (index for index, line in enumerate(lines) if label_pattern.match(line)),
        None,
    )

    if label_index is None:
        raise ValueError(f"Could not locate loop label '{label}:' in baseline source.")

    loop_index = next(
        (
            index
            for index in range(label_index + 1, min(len(lines), label_index + 8))
            if re.search(r"\b(for|while)\s*\(", lines[index])
        ),
        None,
    )

    if loop_index is None:
        raise ValueError(f"Found label '{label}:' but not its loop statement.")

    brace_depth = 0
    body_started = False
    end_index = loop_index

    for index in range(loop_index, len(lines)):
        brace_depth += lines[index].count("{")
        if "{" in lines[index]:
            body_started = True
        brace_depth -= lines[index].count("}")
        end_index = index
        if body_started and brace_depth == 0:
            break

    if not body_started:
        end_index = loop_index

    return label_index, end_index


def map_target_to_source(
    config: dict[str, Any], repo_root: Path, diagnosis_path: Path
) -> Path:
    with diagnosis_path.open("r", encoding="utf-8") as file:
        diagnosis = json.load(file)

    target = select_recommended_target(diagnosis)
    target_name = get_target_name(target)
    loop_label = infer_loop_label(target_name)

    source_path = repo_root / config["baseline"]["source"]
    lines = source_path.read_text(encoding="utf-8").splitlines()
    start_index, end_index = find_labelled_loop(lines, loop_label)

    context_start = max(0, start_index - 3)
    context_end = min(len(lines), end_index + 4)
    source_excerpt = "\n".join(
        f"{index + 1:4d}: {lines[index]}"
        for index in range(context_start, context_end)
    )

    output_dir = repo_root / config["output_dir"]
    output_path = output_dir / "baseline_source_target.json"
    record = {
        "target_name": target_name,
        "loop_label": loop_label,
        "source_file": str(source_path.relative_to(repo_root)),
        "label_line": start_index + 1,
        "region_start_line": start_index + 1,
        "region_end_line": end_index + 1,
        "diagnosis": target,
        "source_excerpt": source_excerpt,
    }
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print("\nSource target mapping")
    print(f"Selected target: {target_name}")
    print(f"Loop label: {loop_label}")
    print(f"Source: {source_path.relative_to(repo_root)}")
    print(f"Lines: {start_index + 1}-{end_index + 1}")
    print("\nRelevant source excerpt")
    print(source_excerpt)

    return output_path


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
    source_target_path = map_target_to_source(config, repo_root, diagnosis_path)

    print("\nStage complete")
    print("Validated the existing baseline synthesis reports.")
    print("Ran the hierarchical bottleneck diagnosis.")
    print("Mapped the selected target back to the baseline source.")
    print(f"Diagnosis: {diagnosis_path.relative_to(repo_root)}")
    print(f"Source target: {source_target_path.relative_to(repo_root)}")
    print("No Vitis synthesis, LLM call, or candidate modification was performed.")


if __name__ == "__main__":
    main()
