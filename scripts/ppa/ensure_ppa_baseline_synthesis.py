#!/usr/bin/env python3

"""Ensure an isolated baseline Vitis HLS synthesis project exists for a PPA run."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIX_PATTERN = r"(?:c|cc|cpp|cxx|h|hpp)"


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


def quote_tcl_path(path: Path) -> str:
    return f'"{path.resolve().as_posix()}"'


def replace_final_file(command: str, absolute_path: Path) -> str:
    match = re.search(rf'([^\s{{}}"]+\.{SOURCE_SUFFIX_PATTERN})\s*$', command)
    if match is None:
        raise ValueError(f"Could not parse add_files path from: {command}")
    return command[: match.start()] + quote_tcl_path(absolute_path)


def absolutize_include_flags(command: str, tcl_dir: Path) -> str:
    """Make every -I path independent of Vitis project and launch directories."""

    def replacement(match: re.Match[str]) -> str:
        raw = match.group(1).strip('"{}')
        include = Path(raw)
        if not include.is_absolute():
            include = tcl_dir / include
        return "-I" + include.resolve().as_posix()

    return re.sub(r"-I([^\s\"]+)", replacement, command)


def absolute_auxiliary_commands(lines: list[str], tcl_dir: Path) -> list[str]:
    commands: list[str] = []
    for line in lines:
        command = line.strip()
        if not re.match(r"^add_files\b", command):
            continue
        if "-tb" not in command and not re.search(r"\.(?:h|hpp)(?:\s|$)", command):
            continue

        match = re.search(rf'([^\s{{}}"]+\.{SOURCE_SUFFIX_PATTERN})\s*$', command)
        if match is None:
            raise ValueError(f"Could not parse auxiliary add_files path from: {command}")
        source = Path(match.group(1))
        if not source.is_absolute():
            source = tcl_dir / source
        command = replace_final_file(command, source)
        command = absolutize_include_flags(command, tcl_dir)
        commands.append(command)
    return commands


def make_tcl(
    baseline_tcl: Path,
    baseline_source: Path,
    project_dir: Path,
) -> tuple[str, str, list[str]]:
    lines = baseline_tcl.read_text(encoding="utf-8").splitlines()
    set_top = first_command(lines, r"^set_top\b", "set_top")
    open_solution = first_command(lines, r"^open_solution(?:\s+-reset)?\b", "open_solution")
    set_part = first_command(lines, r"^set_part\b", "set_part")
    create_clock = first_command(lines, r"^create_clock\b", "create_clock")

    tcl_dir = baseline_tcl.parent.resolve()
    design_command = replace_final_file(design_source_command(lines), baseline_source)
    design_command = absolutize_include_flags(design_command, tcl_dir)
    auxiliary_commands = absolute_auxiliary_commands(lines, tcl_dir)

    commands = [
        f"open_project -reset {quote_tcl_path(project_dir)}",
        set_top,
        design_command,
        *auxiliary_commands,
        open_solution,
        set_part,
        create_clock,
        "csim_design",
        "csynth_design",
        "exit",
    ]
    return "\n".join(commands) + "\n", design_command, auxiliary_commands


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
    tcl_text, design_command, auxiliary_commands = make_tcl(
        baseline_tcl, baseline_source, project_dir
    )
    generated_tcl.write_text(tcl_text, encoding="utf-8")

    command = [find_vitis_run(), "--mode", "hls", "--tcl", str(generated_tcl.resolve())]
    print("\nBaseline CSim and synthesis")
    print(f"Source TCL: {baseline_tcl.relative_to(REPO_ROOT)}")
    print(f"Baseline source: {baseline_source.relative_to(REPO_ROOT)}")
    print(f"Design source command: {design_command}")
    for auxiliary in auxiliary_commands:
        print(f"Auxiliary source command: {auxiliary}")
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
