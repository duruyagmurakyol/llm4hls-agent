#!/usr/bin/env python3

"""Run isolated Vitis HLS synthesis for one CSim-validated PPA candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path("/tmp/llm4hls-agent")


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


def top_name_from_command(set_top: str) -> str:
    parts = set_top.split()
    if len(parts) < 2:
        raise ValueError(f"Could not parse top function from: {set_top}")
    return parts[-1].strip("{}\"")


def parse_add_files(command: str) -> tuple[bool, list[str], str]:
    tokens = shlex.split(command)
    if not tokens or tokens[0] != "add_files":
        raise ValueError(f"Not an add_files command: {command}")

    is_testbench = "-tb" in tokens
    source_index = next(
        (
            index
            for index in range(len(tokens) - 1, 0, -1)
            if re.search(r"\.(?:c|cc|cpp|cxx|h|hpp)$", tokens[index], re.IGNORECASE)
        ),
        None,
    )
    if source_index is None:
        raise ValueError(f"Could not parse source path from: {command}")
    return is_testbench, tokens[1:source_index], tokens[source_index]


def absolutise_cflags(options: list[str], tcl_dir: Path) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(options):
        token = options[index]
        if token == "-cflags" and index + 1 < len(options):
            flags = shlex.split(options[index + 1])
            rewritten: list[str] = []
            for flag in flags:
                if flag.startswith("-I") and len(flag) > 2:
                    include = Path(flag[2:])
                    if not include.is_absolute():
                        include = (tcl_dir / include).resolve()
                    rewritten.append(f"-I{include.as_posix()}")
                else:
                    rewritten.append(flag)
            result.extend(["-cflags", " ".join(rewritten)])
            index += 2
            continue
        result.append(token)
        index += 1
    return result


def absolute_add_files(command: str, tcl_dir: Path, replacement: Path | None = None) -> str:
    is_testbench, options, source_text = parse_add_files(command)
    source = replacement.resolve() if replacement is not None else Path(source_text)
    if replacement is None and not source.is_absolute():
        source = (tcl_dir / source).resolve()
    options = absolutise_cflags(options, tcl_dir)

    tokens = ["add_files"]
    if is_testbench and "-tb" not in options:
        tokens.append("-tb")
    tokens.extend(options)
    tokens.append(source.as_posix())
    return " ".join(shlex.quote(token) for token in tokens)


def temporary_project_dir(output_dir: Path, candidate_index: int) -> Path:
    run_key = hashlib.sha256(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return TMP_ROOT / run_key / f"candidate_{candidate_index:03d}_synthesis"


def make_synthesis_tcl(
    baseline_tcl: Path,
    candidate: Path,
    project_dir: Path,
) -> tuple[str, str, list[str], str]:
    source_lines = baseline_tcl.read_text(encoding="utf-8").splitlines()
    tcl_dir = baseline_tcl.parent.resolve()

    set_top = first_command(source_lines, r"^set_top\b", "set_top")
    top_name = top_name_from_command(set_top)
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
            and re.search(r"\.(?:c|cc|cpp|cxx)(?:\s|$)", line.strip())
        ),
        None,
    )
    if design_line is None:
        raise ValueError("Could not find baseline design-source add_files command.")

    header_lines = [
        line.strip()
        for line in source_lines
        if re.match(r"^add_files\b", line.strip())
        and "-tb" not in line
        and re.search(r"\.(?:h|hpp)(?:\s|$)", line.strip())
    ]

    design_command = absolute_add_files(design_line, tcl_dir, candidate)
    header_commands = [absolute_add_files(line, tcl_dir) for line in header_lines]

    canonical = [
        f'open_project -reset "{project_dir.resolve().as_posix()}"',
        set_top,
        design_command,
        *header_commands,
        open_solution,
        set_part,
        create_clock,
        "csynth_design",
        "exit",
    ]
    return "\n".join(canonical) + "\n", design_command, header_commands, top_name


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


def collect_reports(report_dir: Path, top_name: str) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    xml_paths = sorted(report_dir.glob("*_csynth.xml"))
    top_report = report_dir / f"{top_name}_csynth.xml"
    if not top_report.is_file():
        top_report = None

    hierarchy: dict[str, dict[str, Any]] = {}
    for xml_path in xml_paths:
        function_name = xml_path.name.removesuffix("_csynth.xml")
        hierarchy[function_name] = {
            "csynth_xml": str(xml_path),
            "metrics": parse_csynth_xml(xml_path),
        }
    return top_report, hierarchy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated Vitis HLS synthesis for a CSim-validated PPA candidate."
    )
    parser.add_argument("config", type=Path, help="PPA optimisation JSON config")
    parser.add_argument("--candidate-index", type=int, default=1)
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Reuse the existing temporary synthesis project and only re-extract reports.",
    )
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
    project_dir = temporary_project_dir(output_dir, args.candidate_index)
    generated_tcl = synthesis_dir / "run_synthesis.tcl"
    log_path = synthesis_dir / "vitis_synthesis.log"
    report_path = output_dir / f"candidate_{args.candidate_index:03d}_synthesis.json"
    synthesis_dir.mkdir(parents=True, exist_ok=True)

    if not args.extract_only and project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.parent.mkdir(parents=True, exist_ok=True)

    tcl_text, design_command, header_commands, top_name = make_synthesis_tcl(
        baseline_tcl, candidate.resolve(), project_dir
    )
    generated_tcl.write_text(tcl_text, encoding="utf-8")

    completed_returncode: int | None = None
    if not args.extract_only:
        command = [find_vitis_run(), "--mode", "hls", "--tcl", str(generated_tcl.resolve())]
        print("\nCandidate synthesis")
        print(f"Candidate: {candidate.relative_to(REPO_ROOT)}")
        print(f"Temporary Vitis project: {project_dir}")
        print(f"Generated TCL: {generated_tcl.relative_to(REPO_ROOT)}")
        print(f"Design source command: {design_command}")
        for header in header_commands:
            print(f"Auxiliary source command: {header}")
        print("Running Vitis HLS synthesis...")

        completed = subprocess.run(
            command,
            cwd=baseline_tcl.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        completed_returncode = completed.returncode
        log_path.write_text(completed.stdout, encoding="utf-8")
    else:
        print("\nCandidate synthesis report extraction")
        print(f"Temporary Vitis project: {project_dir}")
        print("Reusing existing synthesis results; Vitis will not be run.")

    report_dir = project_dir / "solution1/syn/report"
    top_report: Path | None = None
    hierarchy: dict[str, dict[str, Any]] = {}
    parse_error: str | None = None
    try:
        top_report, hierarchy = collect_reports(report_dir, top_name)
    except (ET.ParseError, OSError, ValueError) as exc:
        parse_error = str(exc)

    metrics = hierarchy.get(top_name, {}).get("metrics", {})
    synthesis_succeeded = top_report is not None
    passed = synthesis_succeeded and (args.extract_only or completed_returncode == 0)

    report = {
        "candidate_index": args.candidate_index,
        "candidate_file": str(candidate.relative_to(REPO_ROOT)),
        "generated_tcl": str(generated_tcl.relative_to(REPO_ROOT)),
        "design_source_command": design_command,
        "auxiliary_source_commands": header_commands,
        "temporary_project_dir": str(project_dir),
        "log_file": str(log_path.relative_to(REPO_ROOT)) if log_path.exists() else None,
        "top_function": top_name,
        "top_csynth_xml": str(top_report) if top_report else None,
        "return_code": completed_returncode,
        "extract_only": args.extract_only,
        "passed": passed,
        "metrics": metrics,
        "hierarchical_reports": hierarchy,
        "parse_error": parse_error,
        "synthesis_run": not args.extract_only,
        "baseline_modified": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Top function: {top_name}")
    print(f"Top CSynth XML found: {top_report is not None}")
    if metrics:
        print("Top-level metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    print(f"Hierarchical reports found: {len(hierarchy)}")
    if parse_error:
        print(f"Metric parse warning: {parse_error}")
    print(f"Report: {report_path.relative_to(REPO_ROOT)}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    print("The baseline source was not modified.")

    if not passed:
        if not args.extract_only and log_path.exists():
            tail = log_path.read_text(encoding="utf-8").splitlines()[-40:]
            if tail:
                print("\nLast log lines")
                print("\n".join(tail))
        raise SystemExit(completed_returncode or 1)


if __name__ == "__main__":
    main()
