#!/usr/bin/env python3

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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_config(config: dict[str, Any], repo_root: Path) -> None:
    for key in ("experiment_name", "benchmark", "baseline", "output_dir", "budget"):
        if key not in config:
            raise ValueError(f"Missing required config field: {key}")

    baseline = config["baseline"]
    for key in ("source", "tcl", "project_dir"):
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
            f"Baseline synthesis project does not exist: {project_dir}\n"
            "This stage reuses existing reports and does not run Vitis."
        )

    reports = sorted(project_dir.rglob("*csynth.xml"))
    if not reports:
        raise FileNotFoundError(f"No synthesis reports found under: {project_dir}")

    output_dir = repo_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_hierarchical_diagnosis.json"

    print("\nBaseline report validation")
    print(f"Synthesis reports found: {len(reports)}")
    for report in reports:
        print(f"  - {report.relative_to(repo_root)}")

    print("\nRunning hierarchical diagnosis...")
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/analyse_hls_hierarchy.py"),
            str(project_dir),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
    )
    if not output_path.is_file():
        raise RuntimeError(f"Diagnosis output was not created: {output_path}")
    return output_path


def select_target(diagnosis: dict[str, Any]) -> dict[str, Any]:
    ranked = diagnosis.get("ranked_targets")
    if isinstance(ranked, list) and ranked and isinstance(ranked[0], dict):
        return ranked[0]
    recommended = diagnosis.get("recommended_focus")
    if isinstance(recommended, dict):
        return recommended
    raise ValueError("No recommended target found in hierarchy diagnosis.")


def target_name(target: dict[str, Any]) -> str:
    for key in ("function", "target", "name", "report_name"):
        value = target.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Selected target has no usable name.")


def infer_loop_label(name: str) -> str:
    match = re.search(r"Pipeline_(.+)$", name)
    return match.group(1) if match else name.split("_")[-1]


def find_labelled_loop(lines: list[str], label: str) -> tuple[int, int]:
    label_pattern = re.compile(rf"^\s*{re.escape(label)}\s*:\s*$")
    label_index = next(
        (index for index, line in enumerate(lines) if label_pattern.match(line)), None
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
        raise ValueError(f"Found '{label}:' but not its loop statement.")

    depth = 0
    started = False
    end_index = loop_index
    for index in range(loop_index, len(lines)):
        depth += lines[index].count("{")
        started = started or "{" in lines[index]
        depth -= lines[index].count("}")
        end_index = index
        if started and depth == 0:
            break
    return label_index, end_index


def map_target_to_source(
    config: dict[str, Any], repo_root: Path, diagnosis_path: Path
) -> Path:
    diagnosis = load_json(diagnosis_path)
    selected = select_target(diagnosis)
    selected_name = target_name(selected)
    loop_label = infer_loop_label(selected_name)

    source_path = repo_root / config["baseline"]["source"]
    lines = source_path.read_text(encoding="utf-8").splitlines()
    start, end = find_labelled_loop(lines, loop_label)
    context_start = max(0, start - 3)
    context_end = min(len(lines), end + 4)
    excerpt = "\n".join(
        f"{index + 1:4d}: {lines[index]}"
        for index in range(context_start, context_end)
    )

    output_path = repo_root / config["output_dir"] / "baseline_source_target.json"
    record = {
        "target_name": selected_name,
        "loop_label": loop_label,
        "source_file": str(source_path.relative_to(repo_root)),
        "label_line": start + 1,
        "region_start_line": start + 1,
        "region_end_line": end + 1,
        "diagnosis": selected,
        "source_excerpt": excerpt,
    }
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print("\nSource target mapping")
    print(f"Selected target: {selected_name}")
    print(f"Loop label: {loop_label}")
    print(f"Source: {source_path.relative_to(repo_root)}")
    print(f"Lines: {start + 1}-{end + 1}")
    print("\nRelevant source excerpt")
    print(excerpt)
    return output_path


def analyse_source_causes(
    config: dict[str, Any], repo_root: Path, source_target_path: Path
) -> Path:
    source_target = load_json(source_target_path)
    source_path = repo_root / source_target["source_file"]
    lines = source_path.read_text(encoding="utf-8").splitlines()
    start = int(source_target["region_start_line"]) - 1
    end = int(source_target["region_end_line"])
    loop_lines = lines[start:end]

    loop_header = next(
        (line for line in loop_lines if re.search(r"\b(for|while)\s*\(", line)), ""
    )
    induction_variables = set(
        re.findall(r"\b(?:int\s+)?([A-Za-z_]\w*)\s*=", loop_header)
    )
    assignment_pattern = re.compile(
        r"^\s*([A-Za-z_]\w*)\s*=\s*\1\s*([+\-*/])\s*.+;\s*$"
    )

    recurrence_variables: list[str] = []
    recurrence_statements: list[str] = []
    for line in loop_lines:
        match = assignment_pattern.match(line)
        if not match or match.group(1) in induction_variables:
            continue
        variable = match.group(1)
        if variable not in recurrence_variables:
            recurrence_variables.append(variable)
            recurrence_statements.append(line.strip())

    arrays = sorted(
        {
            match.group(1)
            for line in loop_lines
            for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\[", line)
        }
    )

    hypotheses: list[dict[str, Any]] = []
    if recurrence_variables:
        hypotheses.append(
            {
                "category": "loop_carried_reduction_recurrence",
                "confidence": 0.88,
                "evidence": {
                    "variables": recurrence_variables,
                    "statements": recurrence_statements,
                },
                "interpretation": (
                    "Accumulator updates depend on values from previous iterations, "
                    "creating loop-carried dependency chains."
                ),
            }
        )
    if any("*" in statement for statement in recurrence_statements):
        hypotheses.append(
            {
                "category": "arithmetic_operator_chain",
                "confidence": 0.70,
                "evidence": {"statements": recurrence_statements},
                "interpretation": "Multiply-accumulate logic may contribute to the critical path.",
            }
        )
    if len(arrays) >= 2:
        hypotheses.append(
            {
                "category": "memory_access_or_port_pressure",
                "confidence": 0.58,
                "evidence": {"arrays": arrays},
                "interpretation": "Memory banking or port availability remains an alternative cause.",
            }
        )
    if not hypotheses:
        hypotheses.append(
            {
                "category": "source_cause_not_identified",
                "confidence": 0.30,
                "evidence": {},
                "interpretation": "No explicit recurrence or memory pattern was identified.",
            }
        )

    output_path = repo_root / config["output_dir"] / "baseline_source_cause.json"
    record = {
        "target_name": source_target["target_name"],
        "loop_label": source_target["loop_label"],
        "source_file": source_target["source_file"],
        "region_start_line": source_target["region_start_line"],
        "region_end_line": source_target["region_end_line"],
        "primary_hypothesis": hypotheses[0],
        "alternative_hypotheses": hypotheses[1:],
        "analysis_scope": "labelled_loop_only",
    }
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print("\nSource-cause analysis")
    print(f"Primary hypothesis: {hypotheses[0]['category']}")
    print(f"Confidence: {hypotheses[0]['confidence']}")
    if recurrence_variables:
        print(f"Recurrence variables: {', '.join(recurrence_variables)}")
    if arrays:
        print(f"Arrays accessed: {', '.join(arrays)}")
    for alternative in hypotheses[1:]:
        print(
            f"Alternative hypothesis: {alternative['category']} "
            f"(confidence={alternative['confidence']})"
        )
    return output_path


def protected_regions(diagnosis: dict[str, Any]) -> list[str]:
    protected: list[str] = []
    for item in diagnosis.get("ranked_targets", []):
        if not isinstance(item, dict):
            continue
        category = item.get("category") or item.get("primary_category")
        stop = item.get("stop_recommended")
        score = item.get("score")
        if category == "near_sequential_lower_bound" or stop is True:
            name = target_name(item)
            protected.append(name)
        elif isinstance(score, (int, float)) and score < 0:
            protected.append(target_name(item))
    return protected


def generate_optimisation_prompt(
    config: dict[str, Any],
    repo_root: Path,
    diagnosis_path: Path,
    source_target_path: Path,
    source_cause_path: Path,
) -> Path:
    diagnosis = load_json(diagnosis_path)
    source_target = load_json(source_target_path)
    source_cause = load_json(source_cause_path)
    source_path = repo_root / source_target["source_file"]
    full_source = source_path.read_text(encoding="utf-8")

    primary = source_cause["primary_hypothesis"]
    alternatives = source_cause.get("alternative_hypotheses", [])
    protected = protected_regions(diagnosis)
    protected_text = "\n".join(f"- {name}" for name in protected) or "- None identified"
    alternative_text = "\n".join(
        f"- {item['category']} (confidence {item['confidence']})"
        for item in alternatives
    ) or "- None"

    prompt = f"""You are optimising an AMD/Xilinx Vitis HLS C++ kernel.

Objective:
Improve latency or initiation interval while avoiding disproportionate LUT, FF, DSP, or BRAM growth. Correctness is mandatory.

Selected target:
- Function/report: {source_target['target_name']}
- Loop label: {source_target['loop_label']}
- Source region: lines {source_target['region_start_line']}-{source_target['region_end_line']}
- Report diagnosis: {source_target['diagnosis'].get('category', 'unknown')}

Primary source-level hypothesis:
- Category: {primary['category']}
- Confidence: {primary['confidence']}
- Interpretation: {primary.get('interpretation', '')}
- Evidence: {json.dumps(primary.get('evidence', {}), indent=2)}

Alternative hypotheses:
{alternative_text}

Protected regions that should not be modified:
{protected_text}

Constraints:
1. Preserve the function signature and algorithmic behaviour.
2. Modify only the selected loop and declarations or final reduction statements directly required by that loop.
3. Do not modify protected regions.
4. Do not remove the existing HLS top directive.
5. Keep all array bounds valid for fixed dimensions A[38][42], x[42], y[42], and tmp[38].
6. Return one complete compilable C++ source file only.
7. Do not include Markdown fences or explanations.
8. Prefer a focused change addressing the primary hypothesis rather than unrelated pragma changes.
9. Avoid aggressive parallelism unless its expected latency benefit justifies resource growth.

Relevant source excerpt:
{source_target['source_excerpt']}

Complete baseline source:
{full_source}
"""

    output_path = repo_root / config["output_dir"] / "candidate_001_prompt.txt"
    output_path.write_text(prompt, encoding="utf-8")
    print("\nConstrained optimisation prompt")
    print(f"Output: {output_path.relative_to(repo_root)}")
    print(f"Characters: {len(prompt)}")
    print("No model was called.")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diagnosis-guided HLS PPA optimisation.")
    parser.add_argument("config", type=Path, help="Path to the optimisation JSON config.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = load_json(args.config.resolve())
    validate_config(config, repo_root)
    print_configuration(config, repo_root)

    diagnosis_path = diagnose_existing_baseline(config, repo_root)
    source_target_path = map_target_to_source(config, repo_root, diagnosis_path)
    source_cause_path = analyse_source_causes(config, repo_root, source_target_path)
    prompt_path = generate_optimisation_prompt(
        config, repo_root, diagnosis_path, source_target_path, source_cause_path
    )

    print("\nStage complete")
    print("Validated existing baseline synthesis reports.")
    print("Ran hierarchical bottleneck diagnosis.")
    print("Mapped the selected target to source.")
    print("Generated source-level cause hypotheses.")
    print("Generated a constrained optimisation prompt.")
    print(f"Diagnosis: {diagnosis_path.relative_to(repo_root)}")
    print(f"Source target: {source_target_path.relative_to(repo_root)}")
    print(f"Source cause: {source_cause_path.relative_to(repo_root)}")
    print(f"Prompt: {prompt_path.relative_to(repo_root)}")
    print("No Vitis synthesis, LLM call, or candidate modification was performed.")


if __name__ == "__main__":
    main()
