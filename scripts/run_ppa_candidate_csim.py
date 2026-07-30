#!/usr/bin/env python3

"""Run CSim for one generated PPA candidate without modifying the baseline."""

from __future__ import annotations

import argparse
import json
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


def make_csim_tcl(
    baseline_tcl: Path,
    candidate: Path,
    project_dir: Path,
) -> str:
    """Build a clean CSim-only TCL while preserving baseline settings and testbench lines."""

    baseline_lines = baseline_tcl.read_text(encoding="utf-8").splitlines()
    generated: list[str] = [
        f"open_project -reset {{{project_dir.as_posix()}}}",
        f"add_files {{{candidate.as_posix()}}}",
    ]

    saw_top = False
    saw_solution = False
    saw_part = False
    saw_clock = False
    testbench_count = 0

    for line in baseline_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if re.match(r"^open_project\b", stripped):
            continue
        if re.match(r"^add_files\b", stripped):
            if re.search(r"(?:^|\s)-tb(?:\s|$)", stripped):
                generated.append(line)
                testbench_count += 1
            continue
        if re.match(r"^(csim_design|csynth_design|cosim_design|export_design|exit)\b", stripped):
            continue

        generated.append(line)
        saw_top = saw_top or bool(re.match(r"^set_top\b", stripped))
        saw_solution = saw_solution or bool(re.match(r"^open_solution\b", stripped))
        saw_part = saw_part or bool(re.match(r"^set_part\b", stripped))
        saw_clock = saw_clock or bool(re.match(r"^create_clock\b", stripped))

    missing = [
        name
        for name, present in (
            ("set_top", saw_top),
            ("open_solution", saw_solution),
            ("set_part", saw_part),
            ("create_clock", saw_clock),
            ("testbench add_files -tb", testbench_count > 0),
        )
        if not present
    ]
    if missing:
        raise ValueError(
            "Baseline TCL is missing required CSim settings: " + ", ".join(missing)
        )

    generated.extend(["csim_design", "exit"])
    tcl = "\n".join(generated) + "\n"

    candidate_line = f"add_files {{{candidate.as_posix()}}}"
    if tcl.count(candidate_line) != 1:
        raise RuntimeError("Generated TCL does not contain exactly one candidate add_files line.")
    design_add_files = [
        line
        for line in generated
        if re.match(r"^\s*add_files\b", line) and "-tb" not in line
    ]
    if design_add_files != [candidate_line]:
        raise RuntimeError(
            "Generated TCL contains an unexpected design-source add_files command: "
            f"{design_add_files}"
        )

    return tcl


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
    validation_record = load_json(validation)
    if validation_record.get("passed") is not True:
        raise RuntimeError("Static validation did not pass; refusing to run CSim.")

    baseline_tcl = REPO_ROOT / config["baseline"]["tcl"]
    if not baseline_tcl.is_file():
        raise FileNotFoundError(f"Baseline TCL not found: {baseline_tcl}")

    csim_dir = output_dir / f"candidate_{args.candidate_index:03d}_csim"
    project_dir = csim_dir / "project"
    generated_tcl = csim_dir / "run_csim.tcl"
    log_path = csim_dir / "vitis_csim.log"
    report_path = output_dir / f"candidate_{args.candidate_index:03d}_csim_validation.json"
    csim_dir.mkdir(parents=True, exist_ok=True)

    tcl_text = make_csim_tcl(baseline_tcl, candidate, project_dir)
    generated_tcl.write_text(tcl_text, encoding="utf-8")

    command = [find_vitis_run(), "--mode", "hls", "--tcl", str(generated_tcl)]
    print("\nCandidate CSim validation")
    print(f"Candidate: {candidate.relative_to(REPO_ROOT)}")
    print(f"Generated TCL: {generated_tcl.relative_to(REPO_ROOT)}")
    print(f"Design source command: add_files {{{candidate.as_posix()}}}")
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

    candidate_was_compiled = candidate.name in completed.stdout
    passed = completed.returncode == 0 and candidate_was_compiled
    report = {
        "candidate_index": args.candidate_index,
        "candidate_file": str(candidate.relative_to(REPO_ROOT)),
        "generated_tcl": str(generated_tcl.relative_to(REPO_ROOT)),
        "project_dir": str(project_dir.relative_to(REPO_ROOT)),
        "log_file": str(log_path.relative_to(REPO_ROOT)),
        "return_code": completed.returncode,
        "candidate_was_compiled": candidate_was_compiled,
        "passed": passed,
        "synthesis_run": False,
        "baseline_modified": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Return code: {completed.returncode}")
    print(f"Candidate compiled: {candidate_was_compiled}")
    print(f"Log: {log_path.relative_to(REPO_ROOT)}")
    print(f"Report: {report_path.relative_to(REPO_ROOT)}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    print("No synthesis was run and the baseline source was not modified.")

    if not passed:
        tail = completed.stdout.splitlines()[-30:]
        if tail:
            print("\nLast log lines")
            print("\n".join(tail))
        raise SystemExit(completed.returncode or 1)


if __name__ == "__main__":
    main()
