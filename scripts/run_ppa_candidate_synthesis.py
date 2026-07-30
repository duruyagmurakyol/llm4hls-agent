#!/usr/bin/env python3

"""Run isolated Vitis HLS synthesis for one CSim-validated PPA candidate."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_vitis_run() -> str:
    executable = shutil.which("vitis-run")
    if executable:
        return executable
    raise RuntimeError(
        "vitis-run was not found in PATH. Source the AMD/Xilinx settings64.sh file first."
    )


def first_command(lines: list[str], pattern: str, description: str) -> str:
    command = next((line.strip() for line in lines if re.match(pattern, line.strip())), None)
    if command is None:
        raise ValueError(f"Could not find {description} in baseline TCL.")
    return command


def replace_design_source(design_line: str, candidate_relative: str) -> str:
    source_match = re.search(r"([^\s{}\"]+\.(?:c|cc|cpp))\s*$", design_line)
    if source_match is None:
        raise ValueError(f"Could not parse design source path from: {design_line}")
    return design_line[: source_match.start()] + candidate_relative


def make_synthesis_tcl(
    baseline_tcl: Path,
    candidate: Path,
    project_name: str,
) -> tuple[str, str]:
    """Mirror the proven-working baseline topology and run synthesis only."""
    source_lines = baseline_tcl.read_text(encoding="utf-8").splitlines()
    tcl_dir = baseline_tcl.parent.resolve()

    set_top = first_command(source_lines, r"^set_top\b", "set_top")
    open_solution = first_command(
        source_lines, r"^open_solution(?:\s+-reset)?\b", "open_solution"
    )
    set_part = first_command(source_lines, r"^set_part\b", "set_part")
    create_clock = first_command(source_lines, r"^create_clock\b", "create_clock")

    design_line = next(
        (
            line.strip()
            for line in source_lines
            if re.match(r"^add_files\b", line.strip())
            and "-tb" not in line
            and re.search(r"\.(?:c|cc|cpp)(?:\s|$)", line.strip())
        ),
        None,
    )
    if design_line is None:
        raise ValueError("Could not find baseline design-source add_files command.")

    header_commands = [
        line.strip()
        for line in source_lines
        if re.match(r"^add_files\b", line.strip())
        and "-tb" not in line
        and re.search(r"\.(?:h|hpp)(?:\s|$)", line.strip())
    ]

    candidate_relative = Path(os.path.relpath(candidate.resolve(), tcl_dir)).as_posix()
    design_command = replace_design_source(design_line, candidate_relative)

    canonical = [
        f"open_project -reset {project_name}",
        set_top,
        design_command,
        *header_commands,
        open_solution,
        set_part,
        create_clock,
        "csynth_design",
        "exit",
    ]
    return "\n".join(canonical) + "\n", design_command


def text_at(root: ET.Element, path: str) -> str | None:
    node = root.find(path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def as_number(value: str | None) -> int | float | None:
    if value is None or value in {"", "NA", "N/A"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def parse_csynth_xml(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    return {
        "clock_period_ns": as_number(text_at(root, ".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")),
        "latency_best_cycles": as_number(text_at(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency")),
        "latency_average_cycles": as_number(text_at(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency")),
        "latency_worst_cycles": as_number(text_at(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency")),
        "interval_min_cycles": as_number(text_at(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Interval-min")),
        "interval_max_cycles": as_number(text_at(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Interval-max")),
        "resources_lut_used": as_number(text_at(root, ".//AreaEstimates/Resources/LUT")),
        "resources_ff_used": as_number(text_at(root, ".//AreaEstimates/Resources/FF")),
        "resources_dsp_used": as_number(text_at(root, ".//AreaEstimates/Resources/DSP")),
        "resources_bram_used": as_number(text_at(root, ".//AreaEstimates/Resources/BRAM_18K")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated Vitis HLS synthesis for a CSim-validated PPA candidate."
    )
    parser.add_argument("config", type=Path, help="PPA optimisation JSON config")
    parser.add_argument("--candidate-index", type=int, default=1)
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    candidate = output_dir / f"candidate_{args.candidate_index:03d}.cpp"
    csim_report = output_dir / f"candidate_{args.candidate_index:03d}_csim_validation.json"

    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate not found: {candidate}")
    if not csim_report.is_file():
        raise FileNotFoundError(
            f"CSim report not found: {csim_report}\n"
            "Run scripts/run_ppa_candidate_csim.py first."
        )
    if load_json(csim_report).get("passed") is not True:
        raise RuntimeError("Candidate CSim did not pass; refusing to run synthesis.")

    baseline_tcl = REPO_ROOT / config["baseline"]["tcl"]
    if not baseline_tcl.is_file():
        raise FileNotFoundError(f"Baseline TCL not found: {baseline_tcl}")

    synthesis_dir = output_dir / f"candidate_{args.candidate_index:03d}_synthesis"
    generated_tcl = synthesis_dir / "run_synthesis.tcl"
    log_path = synthesis_dir / "vitis_synthesis.log"
    report_path = output_dir / f"candidate_{args.candidate_index:03d}_synthesis.json"
    synthesis_dir.mkdir(parents=True, exist_ok=True)

    project_name = f"atax_candidate_{args.candidate_index:03d}_synthesis_project"
    project_dir = baseline_tcl.parent / project_name
    tcl_text, design_command = make_synthesis_tcl(
        baseline_tcl, candidate.resolve(), project_name
    )
    generated_tcl.write_text(tcl_text, encoding="utf-8")

    command = [find_vitis_run(), "--mode", "hls", "--tcl", str(generated_tcl.resolve())]
    print("\nCandidate synthesis")
    print(f"Candidate: {candidate.relative_to(REPO_ROOT)}")
    print(f"Isolated project: {project_dir.relative_to(REPO_ROOT)}")
    print(f"Generated TCL: {generated_tcl.relative_to(REPO_ROOT)}")
    print(f"Design source command: {design_command}")
    print("Running Vitis HLS synthesis...")

    completed = subprocess.run(
        command,
        cwd=baseline_tcl.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")

    xml_candidates = sorted(project_dir.glob("solution1/syn/report/*_csynth.xml"))
    xml_report = xml_candidates[0] if xml_candidates else None
    metrics: dict[str, Any] = {}
    parse_error: str | None = None
    if xml_report is not None:
        try:
            metrics = parse_csynth_xml(xml_report)
        except (ET.ParseError, OSError, ValueError) as exc:
            parse_error = str(exc)

    passed = completed.returncode == 0 and xml_report is not None
    report = {
        "candidate_index": args.candidate_index,
        "candidate_file": str(candidate.relative_to(REPO_ROOT)),
        "generated_tcl": str(generated_tcl.relative_to(REPO_ROOT)),
        "design_source_command": design_command,
        "project_dir": str(project_dir.relative_to(REPO_ROOT)),
        "log_file": str(log_path.relative_to(REPO_ROOT)),
        "csynth_xml": str(xml_report.relative_to(REPO_ROOT)) if xml_report else None,
        "return_code": completed.returncode,
        "passed": passed,
        "metrics": metrics,
        "parse_error": parse_error,
        "synthesis_run": True,
        "baseline_modified": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Return code: {completed.returncode}")
    print(f"CSynth XML found: {xml_report is not None}")
    if metrics:
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    if parse_error:
        print(f"Metric parse warning: {parse_error}")
    print(f"Log: {log_path.relative_to(REPO_ROOT)}")
    print(f"Report: {report_path.relative_to(REPO_ROOT)}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    print("The baseline source was not modified.")

    if not passed:
        tail = completed.stdout.splitlines()[-40:]
        if tail:
            print("\nLast log lines")
            print("\n".join(tail))
        raise SystemExit(completed.returncode or 1)


if __name__ == "__main__":
    main()
