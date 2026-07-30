#!/usr/bin/env python3

"""Run isolated Vitis HLS CSim for one generated PPA candidate."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
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
    """Preserve the known-working design flags and replace only the source path."""
    source_match = re.search(r"([^\s{}\"]+\.(?:c|cc|cpp))\s*$", design_line)
    if source_match is None:
        raise ValueError(f"Could not parse design source path from: {design_line}")
    return design_line[: source_match.start()] + candidate_relative


def make_csim_tcl(
    baseline_tcl: Path,
    candidate: Path,
    project_name: str,
) -> tuple[str, str]:
    """Mirror the proven-working baseline TCL topology exactly."""
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
    testbench_commands = [
        line.strip()
        for line in source_lines
        if re.match(r"^add_files\b", line.strip()) and "-tb" in line
    ]
    if not testbench_commands:
        raise ValueError("Could not find baseline testbench add_files command.")

    baseline_csim = next(
        (line.strip() for line in source_lines if re.match(r"^csim_design\b", line.strip())),
        "csim_design",
    )

    candidate_relative = Path(os.path.relpath(candidate.resolve(), tcl_dir)).as_posix()
    design_command = replace_design_source(design_line, candidate_relative)

    canonical = [
        f"open_project -reset {project_name}",
        set_top,
        design_command,
        *header_commands,
        *testbench_commands,
        open_solution,
        set_part,
        create_clock,
        baseline_csim,
        "exit",
    ]
    return "\n".join(canonical) + "\n", design_command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated Vitis HLS CSim for a PPA candidate."
    )
    parser.add_argument("config", type=Path, help="PPA optimisation JSON config")
    parser.add_argument("--candidate-index", type=int, default=1)
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    candidate = output_dir / f"candidate_{args.candidate_index:03d}.cpp"
    validation = output_dir / f"candidate_{args.candidate_index:03d}_static_validation.json"

    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate not found: {candidate}")
    if not validation.is_file():
        raise FileNotFoundError(
            f"Static validation report not found: {validation}\n"
            "Run scripts/validate_ppa_candidate.py first."
        )
    if load_json(validation).get("passed") is not True:
        raise RuntimeError("Static validation did not pass; refusing to run CSim.")

    baseline_tcl = REPO_ROOT / config["baseline"]["tcl"]
    if not baseline_tcl.is_file():
        raise FileNotFoundError(f"Baseline TCL not found: {baseline_tcl}")

    csim_dir = output_dir / f"candidate_{args.candidate_index:03d}_csim"
    generated_tcl = csim_dir / "run_csim.tcl"
    log_path = csim_dir / "vitis_csim.log"
    report_path = output_dir / f"candidate_{args.candidate_index:03d}_csim_validation.json"
    csim_dir.mkdir(parents=True, exist_ok=True)

    project_name = f"atax_candidate_{args.candidate_index:03d}_csim_project"
    project_dir = baseline_tcl.parent / project_name
    tcl_text, design_command = make_csim_tcl(
        baseline_tcl, candidate.resolve(), project_name
    )
    generated_tcl.write_text(tcl_text, encoding="utf-8")

    command = [find_vitis_run(), "--mode", "hls", "--tcl", str(generated_tcl.resolve())]
    print("\nCandidate CSim validation")
    print(f"Candidate: {candidate.relative_to(REPO_ROOT)}")
    print(f"Isolated project: {project_dir.relative_to(REPO_ROOT)}")
    print(f"Generated TCL: {generated_tcl.relative_to(REPO_ROOT)}")
    print(f"Design source command: {design_command}")
    print("Running Vitis HLS CSim only...")

    completed = subprocess.run(
        command,
        cwd=baseline_tcl.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")

    candidate_compile_pattern = re.compile(
        rf"Compiling\s+.*{re.escape(candidate.name)}\b", re.IGNORECASE
    )
    candidate_compiled = bool(candidate_compile_pattern.search(completed.stdout))
    passed = completed.returncode == 0 and candidate_compiled

    report = {
        "candidate_index": args.candidate_index,
        "candidate_file": str(candidate.relative_to(REPO_ROOT)),
        "generated_tcl": str(generated_tcl.relative_to(REPO_ROOT)),
        "design_source_command": design_command,
        "project_dir": str(project_dir.relative_to(REPO_ROOT)),
        "log_file": str(log_path.relative_to(REPO_ROOT)),
        "return_code": completed.returncode,
        "candidate_compiled": candidate_compiled,
        "passed": passed,
        "synthesis_run": False,
        "baseline_modified": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Return code: {completed.returncode}")
    print(f"Candidate compiled: {candidate_compiled}")
    print(f"Log: {log_path.relative_to(REPO_ROOT)}")
    print(f"Report: {report_path.relative_to(REPO_ROOT)}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    print("No synthesis was run and the baseline source was not modified.")

    if not passed:
        tail = completed.stdout.splitlines()[-35:]
        if tail:
            print("\nLast log lines")
            print("\n".join(tail))
        raise SystemExit(completed.returncode or 1)


if __name__ == "__main__":
    main()
