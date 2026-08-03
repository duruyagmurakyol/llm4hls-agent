"""Structured Vitis HLS C/RTL co-simulation adapter."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from agent.config import TaskManifest
from agent.tools.command_runner import CommandResult
from agent.tools.reports import write_json
from agent.tools.synthesis import (
    TMP_ROOT,
    _candidate_hash,
    _display_path,
    _resolve,
    _run_vitis,
    _tcl_parts,
    _timeout,
)
from agent.tools.validation import extract_evidence

DEFAULT_COSIM_TIMEOUT_SECONDS = 900


def _failure_class(completed: CommandResult, reports: list[Path]) -> str:
    if completed.timed_out:
        return "cosim_timeout"

    lower = completed.output.lower()
    if "deadlock" in lower or "no progress" in lower:
        return "cosim_deadlock"
    if any(
        token in lower
        for token in (
            "failed to compile",
            "compilation failed",
            "compile error",
            "undefined reference",
            "cannot find",
        )
    ):
        return "cosim_compile"
    if (
        "fail index=" in lower
        or "mismatch" in lower
        or ("expected=" in lower and "actual=" in lower)
        or "simulation failed" in lower
    ):
        return "cosim_mismatch"
    if completed.passed and not reports:
        return "missing_cosim_report"
    return "cosim_failed"


def run_cosim(task: TaskManifest, candidate: Path) -> dict[str, Any]:
    """Run C/RTL co-simulation for one candidate."""
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate not found: {candidate}")

    build_files = task.data["artifacts"].get("build_files", [])
    if not build_files:
        raise ValueError("Task manifest must define artifacts.build_files")

    build_file = _resolve(build_files[0])
    parts, design, auxiliaries, top_name = _tcl_parts(build_file, candidate, True)
    set_top, open_solution, set_part, create_clock = parts
    digest = _candidate_hash(candidate)
    output_dir = _resolve(task.output_dir)
    run_dir = output_dir / "cosim" / digest[:12]
    project_dir = TMP_ROOT / digest[:12] / "cosim"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(project_dir, ignore_errors=True)
    project_dir.parent.mkdir(parents=True, exist_ok=True)

    generated_tcl = run_dir / "run_cosim.tcl"
    generated_tcl.write_text(
        "\n".join(
            [
                f'open_project -reset "{project_dir.resolve().as_posix()}"',
                set_top,
                design,
                *auxiliaries,
                open_solution,
                set_part,
                create_clock,
                "csynth_design",
                "cosim_design",
                "exit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    log_path = run_dir / "vitis_cosim.log"
    completed = _run_vitis(
        generated_tcl,
        build_file.parent,
        log_path,
        _timeout(task.data, "cosim_seconds", DEFAULT_COSIM_TIMEOUT_SECONDS),
    )

    source_report_dir = project_dir / "solution1/sim/report"
    saved_report_dir = run_dir / "reports"
    shutil.rmtree(saved_report_dir, ignore_errors=True)
    if source_report_dir.is_dir():
        shutil.copytree(source_report_dir, saved_report_dir)
    reports = sorted(path for path in saved_report_dir.rglob("*") if path.is_file())

    passed = completed.passed and bool(reports)
    failure_class = "none" if passed else _failure_class(completed, reports)
    evidence = [] if passed else extract_evidence(completed.output)
    if failure_class == "missing_cosim_report":
        evidence = [f"No co-simulation report was generated under {source_report_dir}"]

    report = {
        "passed": passed,
        "timed_out": completed.timed_out,
        "return_code": completed.return_code,
        "failure_class": failure_class,
        "evidence": evidence,
        "command": list(completed.command),
        "duration_seconds": completed.elapsed_seconds,
        "timeout_seconds": completed.timeout_seconds,
        "log_path": _display_path(log_path),
        "candidate_hash": digest,
        "candidate_file": _display_path(candidate),
        "generated_tcl": _display_path(generated_tcl),
        "project_dir": str(project_dir),
        "top_function": top_name,
        "report_dir": _display_path(saved_report_dir),
        "reports": [_display_path(path) for path in reports],
        "cosim_run": True,
        "baseline_modified": False,
    }
    write_json(run_dir / "result.json", report)
    return report
