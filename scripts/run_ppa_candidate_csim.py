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


def replace_add_files_source(tcl: str, baseline_source: Path, candidate: Path) -> str:
    baseline_variants = {
        str(baseline_source),
        baseline_source.as_posix(),
        baseline_source.name,
    }
    updated = tcl
    for value in sorted(baseline_variants, key=len, reverse=True):
        updated = updated.replace(value, candidate.as_posix())

    if candidate.as_posix() in updated:
        return updated

    lines = updated.splitlines()
    source_line_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^\s*add_files\s+", line)
            and "-tb" not in line
            and Path(line.strip().split()[-1].strip("{}\"")).suffix in {".c", ".cc", ".cpp"}
        ),
        None,
    )
    if source_line_index is None:
        raise ValueError("Could not identify the design-source add_files line in baseline TCL.")

    lines[source_line_index] = f"add_files {{{candidate.as_posix()}}}"
    return "\n".join(lines) + "\n"


def make_csim_tcl(
    baseline_tcl: Path,
    baseline_source: Path,
    candidate: Path,
    project_dir: Path,
) -> str:
    tcl = baseline_tcl.read_text(encoding="utf-8")
    tcl = replace_add_files_source(tcl, baseline_source, candidate)

    project_pattern = re.compile(r"^\s*open_project(?:\s+-reset)?\s+.+$", re.MULTILINE)
    replacement = f"open_project -reset {{{project_dir.as_posix()}}}"
    if not project_pattern.search(tcl):
        raise ValueError("Could not find open_project in baseline TCL.")
    tcl = project_pattern.sub(replacement, tcl, count=1)

    filtered: list[str] = []
    for line in tcl.splitlines():
        stripped = line.strip()
        if re.match(r"^(csynth_design|cosim_design|export_design)\b", stripped):
            continue
        filtered.append(line)

    if not any(re.match(r"^\s*csim_design\b", line) for line in filtered):
        insert_index = next(
            (index for index, line in enumerate(filtered) if line.strip() == "exit"),
            len(filtered),
        )
        filtered.insert(insert_index, "csim_design")

    return "\n".join(filtered).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated Vitis HLS CSim for a PPA candidate.")
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

    baseline_source = REPO_ROOT / config["baseline"]["source"]
    baseline_tcl = REPO_ROOT / config["baseline"]["tcl"]
    if not baseline_tcl.is_file():
        raise FileNotFoundError(f"Baseline TCL not found: {baseline_tcl}")

    csim_dir = output_dir / f"candidate_{args.candidate_index:03d}_csim"
    project_dir = csim_dir / "project"
    generated_tcl = csim_dir / "run_csim.tcl"
    log_path = csim_dir / "vitis_csim.log"
    report_path = output_dir / f"candidate_{args.candidate_index:03d}_csim_validation.json"
    csim_dir.mkdir(parents=True, exist_ok=True)

    generated_tcl.write_text(
        make_csim_tcl(baseline_tcl, baseline_source, candidate, project_dir),
        encoding="utf-8",
    )

    command = [find_vitis_run(), "--mode", "hls", "--tcl", str(generated_tcl)]
    print("\nCandidate CSim validation")
    print(f"Candidate: {candidate.relative_to(REPO_ROOT)}")
    print(f"Generated TCL: {generated_tcl.relative_to(REPO_ROOT)}")
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

    passed = completed.returncode == 0
    report = {
        "candidate_index": args.candidate_index,
        "candidate_file": str(candidate.relative_to(REPO_ROOT)),
        "generated_tcl": str(generated_tcl.relative_to(REPO_ROOT)),
        "project_dir": str(project_dir.relative_to(REPO_ROOT)),
        "log_file": str(log_path.relative_to(REPO_ROOT)),
        "return_code": completed.returncode,
        "passed": passed,
        "synthesis_run": False,
        "baseline_modified": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Return code: {completed.returncode}")
    print(f"Log: {log_path.relative_to(REPO_ROOT)}")
    print(f"Report: {report_path.relative_to(REPO_ROOT)}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    print("No synthesis was run and the baseline source was not modified.")

    if not passed:
        tail = completed.stdout.splitlines()[-25:]
        if tail:
            print("\nLast log lines")
            print("\n".join(tail))
        raise SystemExit(completed.returncode or 1)


if __name__ == "__main__":
    main()
