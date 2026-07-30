#!/usr/bin/env python3

"""Ensure an isolated baseline Vitis HLS synthesis project exists for a PPA run."""

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


def design_source_command(lines: list[str]) -> str:
    command = next(
        (
            line.strip()
            for line in lines
            if re.match(r"^add_files\b", line.strip())
            and "-tb" not in line
            and re.search(r"\.(?:c|cc|cpp|cxx)(?:\s|$)", line.strip())
        ),
        None,
    )
    if command is None:
        raise ValueError("Could not find the design-source add_files command in baseline TCL.")
    return command


def replace_design_source(command: str, source_relative: str) -> str:
    match = re.search(r"([^\s{}\"]+\.(?:c|cc|cpp|cxx))\s*$", command)
    if match is None:
        raise ValueError(f"Could not parse design source path from: {command}")
    return command[: match.start()] + source_relative


def auxiliary_source_commands(lines: list[str]) -> list[str]:
    """Keep headers and testbench commands from the proven baseline TCL."""
    return [
        line.strip()
        for line in lines
        if re.match(r"^add_files\b", line.strip())
        and (
            "-tb" in line
            or re.search(r"\.(?:h|hpp)(?:\s|$)", line.strip())
        )
    ]


def make_tcl(baseline_tcl: Path, baseline_source: Path, project_dir: Path) -> tuple[str, str]:
    lines = baseline_tcl.read_text(encoding="utf-8").splitlines()
    set_top = first_command(lines, r"^set_top\b", "set_top")
    open_solution = first_command(lines, r"^open_solution(?:\s+-reset)?\b", "open_solution")
    set_part = first_command(lines, r"^set_part\b", "set_part")
    create_clock = first_command(lines, r"^create_clock\b", "create_clock")

    tcl_dir = baseline_tcl.parent.resolve()
    source_relative = Path(os.path.relpath(baseline_source.resolve(), tcl_dir)).as_posix()
    design_command = replace_design_source(design_source_command(lines), source_relative)

    commands = [
        f'open_project -reset "{project_dir.resolve().as_posix()}"',
        set_top,
        design_command,
        *auxiliary_source_commands(lines),
        open_solution,
        set_part,
        create_clock,
        "csim_design",
        "csynth_design",
        "exit",
    ]
    return "\n".join(commands) + "\n", design_command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline CSim and synthesis only when usable reports are absent."
    )
    parser.add_argument("config", type=Path, help="PPA optimisation JSON config")
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    baseline = config["baseline"]
    baseline_tcl = REPO_ROOT / baseline["tcl"]
    baseline_source = REPO_ROOT / baseline["source"]
    project_dir = REPO_ROOT / baseline["project_dir"]
    output_dir = REPO_ROOT / config["output_dir"]
    run_dir = output_dir / "baseline_run"
    generated_tcl = run_dir / "run_baseline.tcl"
    log_path = run_dir / "vitis_baseline.log"

    if not baseline_tcl.is_file():
        raise FileNotFoundError(f"Baseline TCL not found: {baseline_tcl}")
    if not baseline_source.is_file():
        raise FileNotFoundError(f"Baseline source not found: {baseline_source}")

    existing_reports = sorted(project_dir.rglob("*_csynth.xml")) if project_dir.is_dir() else []
    if existing_reports:
        print("\nBaseline synthesis cache")
        print(f"Project: {project_dir.relative_to(REPO_ROOT)}")
        print(f"Synthesis reports found: {len(existing_reports)}")
        print("Vitis was not run.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    tcl_text, design_command = make_tcl(baseline_tcl, baseline_source, project_dir)
    generated_tcl.write_text(tcl_text, encoding="utf-8")

    command = [find_vitis_run(), "--mode", "hls", "--tcl", str(generated_tcl.resolve())]
    print("\nBaseline CSim and synthesis")
    print(f"Source TCL: {baseline_tcl.relative_to(REPO_ROOT)}")
    print(f"Baseline source: {baseline_source.relative_to(REPO_ROOT)}")
    print(f"Design source command: {design_command}")
    print(f"Isolated project: {project_dir.relative_to(REPO_ROOT)}")
    print(f"Generated TCL: {generated_tcl.relative_to(REPO_ROOT)}")
    print("Running Vitis HLS because no baseline synthesis reports exist...")

    completed = subprocess.run(
        command,
        cwd=baseline_tcl.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")

    reports = sorted(project_dir.rglob("*_csynth.xml")) if project_dir.is_dir() else []
    passed = completed.returncode == 0 and bool(reports)
    print(f"Return code: {completed.returncode}")
    print(f"Synthesis reports found: {len(reports)}")
    print(f"Log: {log_path.relative_to(REPO_ROOT)}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")

    if not passed:
        tail = completed.stdout.splitlines()[-40:]
        if tail:
            print("\nLast log lines")
            print("\n".join(tail))
        raise SystemExit(completed.returncode or 1)


if __name__ == "__main__":
    main()
