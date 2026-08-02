#!/usr/bin/env python3

"""Run correctness-preserving, synthesis-guided vector-add PPA optimisation."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.providers.siliconflow import complete  # noqa: E402

METRICS = (
    "latency_best_cycles",
    "latency_worst_cycles",
    "interval_min_cycles",
    "interval_max_cycles",
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
)


def run(command: list[str], cwd: Path, log: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    log.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + process.stdout
        + "\nSTDERR\n" + process.stderr,
        encoding="utf-8",
    )
    return process


def host_validate(workspace: Path, output_dir: Path, label: str) -> bool:
    compile_result = run(
        [
            "g++", "-std=c++17", "-Isrc", "src/vector_add.cpp",
            "testbench/vector_add_test.cpp", "-o", f"host_test_{label}",
        ],
        workspace,
        output_dir / f"{label}_host_compile.log",
    )
    if compile_result.returncode != 0:
        return False
    execute = run(
        [f"./host_test_{label}"], workspace, output_dir / f"{label}_host_run.log"
    )
    return execute.returncode == 0


def synthesis(workspace: Path, output_dir: Path, label: str) -> tuple[bool, dict[str, int | float | None]]:
    project = workspace / "vector_add_hls"
    if project.exists():
        shutil.rmtree(project)
    process = run(
        ["vitis-run", "--mode", "hls", "--tcl", "run_hls.tcl"],
        workspace,
        output_dir / f"{label}_vitis.log",
    )
    metrics = parse_metrics(workspace)
    (output_dir / f"{label}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return process.returncode == 0 and metrics["latency_best_cycles"] is not None, metrics


def first_number(root: ET.Element, names: tuple[str, ...]) -> int | float | None:
    wanted = {name.lower() for name in names}
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag not in wanted or element.text is None:
            continue
        text = element.text.strip()
        try:
            value = float(text)
            return int(value) if value.is_integer() else value
        except ValueError:
            continue
    return None


def parse_metrics(workspace: Path) -> dict[str, int | float | None]:
    reports = sorted(workspace.glob("vector_add_hls/**/vector_add_csynth.xml"))
    if not reports:
        reports = sorted(workspace.glob("vector_add_hls/**/*csynth.xml"))
    result = {name: None for name in METRICS}
    if not reports:
        return result

    root = ET.parse(reports[-1]).getroot()
    result.update(
        {
            "latency_best_cycles": first_number(root, ("Best-caseLatency", "BestLatency")),
            "latency_worst_cycles": first_number(root, ("Worst-caseLatency", "WorstLatency")),
            "interval_min_cycles": first_number(root, ("Interval-min", "MinInterval")),
            "interval_max_cycles": first_number(root, ("Interval-max", "MaxInterval")),
        }
    )

    resource_aliases = {
        "resources_lut_used": ("LUT",),
        "resources_ff_used": ("FF",),
        "resources_dsp_used": ("DSP", "DSP48E"),
        "resources_bram_used": ("BRAM_18K", "BRAM"),
    }
    for key, aliases in resource_aliases.items():
        result[key] = first_number(root, aliases)
    return result


def extract_cpp(text: str) -> str:
    blocks = re.findall(r"```(?:cpp|c\+\+|c)?\s*(.*?)```", text, flags=re.I | re.S)
    candidate = blocks[-1] if blocks else text
    candidate = re.sub(r"^\s*(?:filename\s*:)?\s*src/vector_add\.cpp\s*$", "", candidate, flags=re.I | re.M)
    candidate = candidate.strip()
    if "```" in candidate or not candidate.startswith("#include"):
        raise ValueError("Model response did not contain clean C++ source")
    return candidate + "\n"


def protected_hashes(workspace: Path, files: list[str]) -> dict[str, str]:
    return {
        relative: hashlib.sha256((workspace / relative).read_bytes()).hexdigest()
        for relative in files
    }


def delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in METRICS:
        before, after = baseline.get(metric), candidate.get(metric)
        result[metric] = None if before is None or after is None else float(after) - float(before)
    return result


def acceptable(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    primary = "latency_best_cycles"
    if baseline.get(primary) is None or candidate.get(primary) is None:
        reasons.append("missing_primary_metric")
    elif float(candidate[primary]) > float(baseline[primary]):
        reasons.append("primary_metric_regressed")

    comparable = [
        metric for metric in METRICS
        if baseline.get(metric) is not None and candidate.get(metric) is not None
    ]
    if not any(float(candidate[m]) < float(baseline[m]) for m in comparable):
        reasons.append("no_metric_improved")
    return not reasons, reasons


def prompt(source: str, metrics: dict[str, Any], previous: list[dict[str, Any]]) -> str:
    history = json.dumps(previous, indent=2) if previous else "None"
    return f"""Optimise this correct Vitis HLS vector-add implementation for hardware PPA.

Constraints:
- Return only the complete contents of src/vector_add.cpp.
- Preserve the exact function signature and integer behaviour.
- Do not change headers, testbench, array size, target device, or clock.
- HLS pragmas and source restructuring are allowed.
- Primary objective: reduce best-case latency cycles.
- Secondary objective: reduce LUT, FF, DSP, and BRAM use without worsening latency.
- Avoid unsafe transformations and preserve synthesizability.

Baseline synthesis metrics:
{json.dumps(metrics, indent=2)}

Previous rejected candidates:
{history}

Current source:
```cpp
{source}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/vector_add_ppa_qwen35.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))

    source_workspace = ROOT / config["workspace"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "results/experiments/vector_add_ppa_qwen35" / timestamp
    workspace = output_dir / "workspace"
    output_dir.mkdir(parents=True)
    shutil.copytree(source_workspace, workspace)

    editable = workspace / config["editable_file"]
    baseline_source = editable.read_text(encoding="utf-8")
    hashes = protected_hashes(workspace, config["protected_files"])

    baseline_host = host_validate(workspace, output_dir, "baseline")
    baseline_synth, baseline_metrics = synthesis(workspace, output_dir, "baseline")
    if not baseline_host or not baseline_synth:
        raise SystemExit(f"Baseline validation/synthesis failed. See {output_dir.relative_to(ROOT)}")

    records: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    synthesis_calls = 1

    for index in range(1, int(config["max_candidates"]) + 1):
        response = complete(
            model=config["model"],
            system_prompt="You are an expert AMD Vitis HLS optimisation agent.",
            user_prompt=prompt(editable.read_text(encoding="utf-8"), baseline_metrics, records),
            temperature=float(config["temperature"]),
            max_tokens=int(config["max_output_tokens"]),
            timeout_seconds=int(config["api_timeout_seconds"]),
            thinking_budget=config.get("thinking_budget"),
        )
        candidate_dir = output_dir / f"candidate_{index}"
        candidate_dir.mkdir()
        (candidate_dir / "raw_response.txt").write_text(response.content, encoding="utf-8")
        try:
            candidate_source = extract_cpp(response.content)
        except ValueError as error:
            records.append({"candidate": index, "status": "parse_failed", "error": str(error)})
            continue

        editable.write_text(candidate_source, encoding="utf-8")
        (candidate_dir / "candidate.cpp").write_text(candidate_source, encoding="utf-8")
        host_ok = host_validate(workspace, candidate_dir, "candidate")
        record: dict[str, Any] = {
            "candidate": index,
            "tokens": response.total_tokens,
            "api_latency_seconds": round(response.latency_seconds, 3),
            "host_passed": host_ok,
        }
        if not host_ok:
            record["status"] = "correctness_rejected"
            records.append(record)
            editable.write_text(baseline_source, encoding="utf-8")
            continue

        if synthesis_calls >= int(config["synthesis_budget"]):
            record["status"] = "synthesis_budget_exhausted"
            records.append(record)
            editable.write_text(baseline_source, encoding="utf-8")
            break

        synth_ok, candidate_metrics = synthesis(workspace, candidate_dir, "candidate")
        synthesis_calls += 1
        record["synthesis_passed"] = synth_ok
        record["metrics"] = candidate_metrics
        record["delta"] = delta(baseline_metrics, candidate_metrics)
        if synth_ok:
            is_acceptable, reasons = acceptable(baseline_metrics, candidate_metrics)
        else:
            is_acceptable, reasons = False, ["synthesis_failed"]
        record["acceptance_reasons"] = reasons
        record["status"] = "accepted" if is_acceptable else "ppa_rejected"
        records.append(record)

        diff = "".join(
            difflib.unified_diff(
                baseline_source.splitlines(True), candidate_source.splitlines(True),
                fromfile="baseline/src/vector_add.cpp", tofile="candidate/src/vector_add.cpp",
            )
        )
        (candidate_dir / "repair.diff").write_text(diff, encoding="utf-8")
        if is_acceptable:
            accepted = record
            break
        editable.write_text(baseline_source, encoding="utf-8")

    unchanged = protected_hashes(workspace, config["protected_files"]) == hashes
    if accepted is None:
        editable.write_text(baseline_source, encoding="utf-8")

    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "timestamp_utc": timestamp,
        "model": config["model"],
        "baseline_host_passed": baseline_host,
        "baseline_synthesis_passed": baseline_synth,
        "baseline_metrics": baseline_metrics,
        "synthesis_calls": synthesis_calls,
        "synthesis_budget": config["synthesis_budget"],
        "candidates_attempted": len(records),
        "accepted": accepted is not None,
        "accepted_candidate": accepted,
        "protected_files_unchanged": unchanged,
        "candidates": records,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("Vector-add PPA experiment")
    print(f"Baseline metrics: {baseline_metrics}")
    print(f"Accepted improvement: {accepted is not None}")
    if accepted:
        print(f"Candidate metrics: {accepted['metrics']}")
        print(f"Metric deltas: {accepted['delta']}")
    print(f"Synthesis calls: {synthesis_calls}/{config['synthesis_budget']}")
    print(f"Protected files unchanged: {unchanged}")
    print(f"Results: {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
