"""Structured Vitis HLS C/RTL co-simulation adapter."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from agent.config import TaskManifest
from agent.failures import classify_failure
from agent.tools.reports import load_json, write_json
from agent.tools.synthesis import (
    REPO_ROOT,
    TMP_ROOT,
    _candidate_hash,
    _display_path,
    _project_key,
    _resolve,
    _run_vitis,
    _tcl_parts,
    _timeout,
)
from agent.tools.validation import extract_evidence

DEFAULT_COSIM_TIMEOUT_SECONDS = 900


def _collect_reports(project_dir: Path, run_dir: Path) -> tuple[Path, list[Path]]:
    source_report_dir = project_dir / "solution1/sim/report"
    saved_report_dir = run_dir / "reports"
    shutil.rmtree(saved_report_dir, ignore_errors=True)
    if source_report_dir.is_dir():
        shutil.copytree(source_report_dir, saved_report_dir)
    reports = sorted(path for path in saved_report_dir.rglob("*") if path.is_file())
    return saved_report_dir, reports


def _classify_cosim(
    *,
    completed: Any,
    project_dir: Path,
    reports: list[Path],
) -> tuple[bool, str, list[str]]:
    passed = completed.passed and bool(reports)
    if passed:
        return True, "none", []
    if completed.passed and not reports:
        return (
            False,
            "tool_report_missing",
            [f"No co-simulation report was generated under {project_dir / 'solution1/sim/report'}"],
        )
    return (
        False,
        classify_failure(
            completed.output,
            stage="cosim",
            timed_out=completed.timed_out,
        ),
        extract_evidence(completed.output),
    )


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
    saved_report_dir, reports = _collect_reports(project_dir, run_dir)
    passed, failure_class, evidence = _classify_cosim(
        completed=completed,
        project_dir=project_dir,
        reports=reports,
    )

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


def run_candidate_cosim(config_path: Any, candidate_index: int = 1) -> dict[str, Any]:
    """Run C/RTL co-simulation for a synthesised optimisation candidate."""
    config = load_json(config_path.resolve())
    output_dir = _resolve(config["output_dir"])
    candidate = output_dir / f"candidate_{candidate_index:03d}.cpp"
    synthesis_report = output_dir / f"candidate_{candidate_index:03d}_synthesis.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate not found: {candidate}")
    if not synthesis_report.is_file() or load_json(synthesis_report).get("passed") is not True:
        raise RuntimeError("Candidate synthesis did not pass; refusing to run co-simulation.")

    baseline_tcl = _resolve(config["baseline"]["tcl"])
    parts, design, auxiliaries, top_name = _tcl_parts(baseline_tcl, candidate, True)
    set_top, open_solution, set_part, create_clock = parts
    digest = _candidate_hash(candidate)
    run_dir = output_dir / f"candidate_{candidate_index:03d}_cosim"
    project_dir = TMP_ROOT / _project_key(output_dir) / f"candidate_{candidate_index:03d}_cosim"
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
        baseline_tcl.parent,
        log_path,
        _timeout(config, "cosim_seconds", DEFAULT_COSIM_TIMEOUT_SECONDS),
    )
    saved_report_dir, reports = _collect_reports(project_dir, run_dir)
    passed, failure_class, evidence = _classify_cosim(
        completed=completed,
        project_dir=project_dir,
        reports=reports,
    )

    report = {
        "candidate_index": candidate_index,
        "candidate_file": _display_path(candidate),
        "candidate_hash": digest,
        "generated_tcl": _display_path(generated_tcl),
        "design_source_command": design,
        "auxiliary_source_commands": auxiliaries,
        "project_dir": str(project_dir),
        "project_storage": "temporary",
        "top_function": top_name,
        "return_code": completed.return_code,
        "timed_out": completed.timed_out,
        "timeout_seconds": completed.timeout_seconds,
        "elapsed_seconds": completed.elapsed_seconds,
        "failure_class": failure_class,
        "evidence": evidence,
        "passed": passed,
        "log_file": _display_path(log_path),
        "report_dir": _display_path(saved_report_dir),
        "reports": [_display_path(path) for path in reports],
        "cosim_run": True,
        "baseline_modified": False,
    }
    write_json(output_dir / f"candidate_{candidate_index:03d}_cosim.json", report)
    return report
