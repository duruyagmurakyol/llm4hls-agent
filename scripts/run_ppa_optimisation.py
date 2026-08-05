#!/usr/bin/env python3

"""Diagnose a baseline HLS project and prepare a benchmark-configured optimisation prompt."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any], repo_root: Path) -> None:
    for key in ("experiment_name", "benchmark", "top_function", "baseline", "output_dir", "budget"):
        if key not in config:
            raise ValueError(f"Missing required config field: {key}")
    for key in ("source", "tcl", "project_dir"):
        if key not in config["baseline"]:
            raise ValueError(f"Missing baseline field: {key}")
    for description, path in {
        "baseline source": repo_root / config["baseline"]["source"],
        "baseline TCL": repo_root / config["baseline"]["tcl"],
        "hierarchical analyser": repo_root / "scripts/analyse_hls_hierarchy.py",
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")


def project_path(config: dict[str, Any], repo_root: Path) -> Path:
    configured = Path(config["baseline"]["project_dir"])
    return configured if configured.is_absolute() else repo_root / configured


def diagnose_existing_baseline(config: dict[str, Any], repo_root: Path) -> Path:
    project_dir = project_path(config, repo_root)
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Baseline synthesis project does not exist: {project_dir}")
    reports = sorted(project_dir.rglob("*csynth.xml"))
    if not reports:
        raise FileNotFoundError(f"No synthesis reports found under: {project_dir}")
    output_dir = repo_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_hierarchical_diagnosis.json"
    subprocess.run([
        sys.executable,
        str(repo_root / "scripts/analyse_hls_hierarchy.py"),
        str(project_dir),
        "--output",
        str(output_path),
    ], cwd=repo_root, check=True)
    return output_path


def target_name(target: dict[str, Any]) -> str:
    for key in ("function", "target", "name", "report_name"):
        value = target.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Selected target has no usable name")


def select_target(diagnosis: dict[str, Any]) -> dict[str, Any]:
    ranked = diagnosis.get("ranked_targets")
    if isinstance(ranked, list) and ranked and isinstance(ranked[0], dict):
        return ranked[0]
    recommended = diagnosis.get("recommended_focus")
    if isinstance(recommended, dict):
        return recommended
    raise ValueError("No recommended target found in hierarchy diagnosis")


def infer_loop_label(name: str) -> str:
    match = re.search(r"Pipeline_(.+)$", name)
    return match.group(1) if match else name.split("_")[-1]


def find_labelled_loop(lines: list[str], label: str) -> tuple[int, int]:
    label_index = next((i for i, line in enumerate(lines) if re.match(rf"^\s*{re.escape(label)}\s*:\s*$", line)), None)
    if label_index is None:
        raise ValueError(f"Could not locate loop label '{label}:' in baseline source")
    loop_index = next((i for i in range(label_index + 1, min(len(lines), label_index + 8)) if re.search(r"\b(for|while)\s*\(", lines[i])), None)
    if loop_index is None:
        raise ValueError(f"Found '{label}:' but not its loop statement")
    depth, started, end_index = 0, False, loop_index
    for index in range(loop_index, len(lines)):
        depth += lines[index].count("{")
        started = started or "{" in lines[index]
        depth -= lines[index].count("}")
        end_index = index
        if started and depth == 0:
            break
    return label_index, end_index


def map_target_to_source(config: dict[str, Any], repo_root: Path, diagnosis_path: Path) -> Path:
    diagnosis = load_json(diagnosis_path)
    selected = select_target(diagnosis)
    selected_name = target_name(selected)
    configured_label = config.get("target_loop_label")
    loop_label = configured_label if isinstance(configured_label, str) and configured_label else infer_loop_label(selected_name)
    source_path = repo_root / config["baseline"]["source"]
    lines = source_path.read_text(encoding="utf-8").splitlines()
    try:
        start, end = find_labelled_loop(lines, loop_label)
    except ValueError:
        loop_index = next(
            (
                index
                for index, line in enumerate(lines)
                if re.search(r"\b(for|while)\s*\(", line)
            ),
            None,
        )
        if loop_index is None:
            raise ValueError(
                f"Could not map diagnosis target '{selected_name}' to a source loop"
            )

        depth = 0
        started = False
        end = loop_index

        for index in range(loop_index, len(lines)):
            depth += lines[index].count("{")
            started = started or "{" in lines[index]
            depth -= lines[index].count("}")
            end = index

            if started and depth == 0:
                break

        start = loop_index
        loop_label = None
    excerpt = "\n".join(f"{i + 1:4d}: {lines[i]}" for i in range(max(0, start - 3), min(len(lines), end + 4)))
    output_path = repo_root / config["output_dir"] / "baseline_source_target.json"
    output_path.write_text(json.dumps({
        "target_name": selected_name,
        "loop_label": loop_label,
        "source_file": str(source_path.relative_to(repo_root)),
        "label_line": start + 1 if loop_label else None,
        "region_start_line": start + 1,
        "region_end_line": end + 1,
        "diagnosis": selected,
        "source_excerpt": excerpt,
    }, indent=2) + "\n", encoding="utf-8")
    return output_path


def analyse_source_causes(config: dict[str, Any], repo_root: Path, source_target_path: Path) -> Path:
    target = load_json(source_target_path)
    source = (repo_root / target["source_file"]).read_text(encoding="utf-8").splitlines()
    region = source[int(target["region_start_line"]) - 1:int(target["region_end_line"])]
    header = next((line for line in region if re.search(r"\b(for|while)\s*\(", line)), "")
    induction = set(re.findall(r"\b(?:int\s+)?([A-Za-z_]\w*)\s*=", header))
    recurrence_variables: list[str] = []
    recurrence_statements: list[str] = []
    for line in region:
        match = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*\1\s*([+\-*/])\s*.+;\s*$", line)
        if match and match.group(1) not in induction and match.group(1) not in recurrence_variables:
            recurrence_variables.append(match.group(1))
            recurrence_statements.append(line.strip())
    arrays = sorted({m.group(1) for line in region for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\[", line)})
    hypotheses: list[dict[str, Any]] = []
    if recurrence_variables:
        hypotheses.append({"category": "loop_carried_reduction_recurrence", "confidence": 0.88, "evidence": {"variables": recurrence_variables, "statements": recurrence_statements}, "interpretation": "Accumulator updates create loop-carried dependency chains."})
    if len(arrays) >= 2:
        hypotheses.append({"category": "memory_access_or_port_pressure", "confidence": 0.70 if not recurrence_variables else 0.58, "evidence": {"arrays": arrays}, "interpretation": "Memory banking or port availability may limit useful parallelism."})
    if not hypotheses:
        hypotheses.append({"category": "source_cause_not_identified", "confidence": 0.30, "evidence": {}, "interpretation": "No explicit recurrence or multi-array pressure was identified."})
    output_path = repo_root / config["output_dir"] / "baseline_source_cause.json"
    output_path.write_text(json.dumps({
        "target_name": target["target_name"],
        "loop_label": target["loop_label"],
        "source_file": target["source_file"],
        "region_start_line": target["region_start_line"],
        "region_end_line": target["region_end_line"],
        "primary_hypothesis": hypotheses[0],
        "alternative_hypotheses": hypotheses[1:],
        "analysis_scope": "labelled_loop_only",
    }, indent=2) + "\n", encoding="utf-8")
    return output_path


def generate_optimisation_prompt(config: dict[str, Any], repo_root: Path, diagnosis_path: Path, source_target_path: Path, source_cause_path: Path) -> Path:
    diagnosis = load_json(diagnosis_path)
    target = load_json(source_target_path)
    cause = load_json(source_cause_path)
    full_source = (repo_root / target["source_file"]).read_text(encoding="utf-8")
    primary = cause["primary_hypothesis"]
    alternatives = cause.get("alternative_hypotheses", [])
    constraints = config.get("prompt_constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("prompt_constraints must be a list of strings")
    base_constraints = [
        f"Preserve the exact {config['top_function']} function signature and algorithmic behaviour.",
        "Modify only the selected loop and declarations directly required by that change.",
        "Do not remove the existing HLS top directive.",
        "Return one complete compilable C++ source file only, without Markdown fences or explanations.",
        "Keep every array access in bounds and preserve all input elements exactly once.",
        "Prefer a focused evidence-driven change over unrelated pragma additions.",
    ]
    numbered = "\n".join(f"{i}. {item}" for i, item in enumerate([*base_constraints, *constraints], 1))
    alternative_text = "\n".join(f"- {item['category']} (confidence {item['confidence']})" for item in alternatives) or "- None"
    prompt = f"""You are optimising an AMD/Xilinx Vitis HLS C++ kernel.

Benchmark: {config['benchmark']}
Top function: {config['top_function']}
Objective: improve latency or initiation interval while avoiding disproportionate LUT, FF, DSP, or BRAM growth. Correctness is mandatory.

Selected target:
- Function/report: {target['target_name']}
- Loop label: {target['loop_label']}
- Source region: lines {target['region_start_line']}-{target['region_end_line']}
- Report diagnosis: {target['diagnosis'].get('category', 'unknown')}

Primary source-level hypothesis:
- Category: {primary['category']}
- Confidence: {primary['confidence']}
- Interpretation: {primary.get('interpretation', '')}
- Evidence: {json.dumps(primary.get('evidence', {}), indent=2)}

Alternative hypotheses:
{alternative_text}

Constraints:
{numbered}

Relevant source excerpt:
{target['source_excerpt']}

Complete baseline source:
{full_source}
"""
    output_path = repo_root / config["output_dir"] / "candidate_001_prompt.txt"
    output_path.write_text(prompt, encoding="utf-8")
    print("\nConstrained optimisation prompt")
    print(f"Benchmark: {config['benchmark']}")
    print(f"Top function: {config['top_function']}")
    print(f"Output: {output_path.relative_to(repo_root)}")
    print("No model was called.")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diagnosis-guided HLS PPA optimisation.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_json(args.config.resolve())
    validate_config(config, repo_root)
    diagnosis = diagnose_existing_baseline(config, repo_root)
    target = map_target_to_source(config, repo_root, diagnosis)
    cause = analyse_source_causes(config, repo_root, target)
    prompt = generate_optimisation_prompt(config, repo_root, diagnosis, target, cause)
    print("\nStage complete")
    print(f"Diagnosis: {diagnosis.relative_to(repo_root)}")
    print(f"Source target: {target.relative_to(repo_root)}")
    print(f"Source cause: {cause.relative_to(repo_root)}")
    print(f"Prompt: {prompt.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
