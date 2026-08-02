"""Portable Vitis HLS CSim, synthesis and report extraction tools."""

from __future__ import annotations

import argparse
import hashlib
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from agent.tools.command_runner import CommandResult, run_command
from agent.tools.reports import load_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
TMP_ROOT = Path(tempfile.gettempdir()) / "llm4hls-agent"
DEFAULT_CSIM_TIMEOUT_SECONDS = 300
DEFAULT_SYNTHESIS_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class VitisResult:
    returncode: int
    output: str
    timed_out: bool
    timeout_seconds: int
    elapsed_seconds: float


def run_synthesis_adapter(command: Sequence[str | Path], *, repository_root: Path) -> CommandResult:
    return run_command(command, cwd=repository_root)


def find_vitis_run() -> str:
    executable = shutil.which("vitis-run")
    if executable:
        return executable
    raise RuntimeError("vitis-run was not found in PATH. Source the AMD/Xilinx settings64.sh file first.")


def _first(lines: list[str], pattern: str, description: str) -> str:
    value = next((line.strip() for line in lines if re.match(pattern, line.strip())), None)
    if value is None:
        raise ValueError(f"Could not find {description} in baseline TCL.")
    return value


def _parse_add_files(command: str) -> tuple[bool, list[str], str]:
    tokens = shlex.split(command)
    if not tokens or tokens[0] != "add_files":
        raise ValueError(f"Not an add_files command: {command}")
    source_index = next(
        (
            i
            for i in range(len(tokens) - 1, 0, -1)
            if re.search(r"\.(?:c|cc|cpp|cxx|h|hpp)$", tokens[i], re.I)
        ),
        None,
    )
    if source_index is None:
        raise ValueError(f"Could not parse source path from: {command}")
    return "-tb" in tokens, tokens[1:source_index], tokens[source_index]


def _absolute_options(options: list[str], tcl_dir: Path) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(options):
        token = options[index]
        if token == "-cflags" and index + 1 < len(options):
            flags = []
            for flag in shlex.split(options[index + 1]):
                if flag.startswith("-I") and len(flag) > 2:
                    include = Path(flag[2:])
                    if not include.is_absolute():
                        include = (tcl_dir / include).resolve()
                    flag = f"-I{include.as_posix()}"
                flags.append(flag)
            result.extend(["-cflags", " ".join(flags)])
            index += 2
        else:
            result.append(token)
            index += 1
    return result


def _absolute_add_files(command: str, tcl_dir: Path, replacement: Path | None = None) -> str:
    is_tb, options, source_text = _parse_add_files(command)
    source = replacement.resolve() if replacement else Path(source_text)
    if replacement is None and not source.is_absolute():
        source = (tcl_dir / source).resolve()
    options = _absolute_options(options, tcl_dir)
    tokens = ["add_files"]
    if is_tb and "-tb" not in options:
        tokens.append("-tb")
    tokens.extend(options)
    tokens.append(source.as_posix())
    return " ".join(shlex.quote(token) for token in tokens)


def _tcl_parts(
    baseline_tcl: Path,
    candidate: Path,
    include_testbench: bool,
) -> tuple[list[str], str, list[str], str]:
    lines = baseline_tcl.read_text(encoding="utf-8").splitlines()
    tcl_dir = baseline_tcl.parent.resolve()
    set_top = _first(lines, r"^set_top\b", "set_top")
    top_name = set_top.split()[-1].strip("{}\"")
    open_solution = _first(lines, r"^open_solution(?:\s+-reset)?\b", "open_solution")
    set_part = _first(lines, r"^set_part\b", "set_part")
    create_clock = _first(lines, r"^create_clock\b", "create_clock")
    design = next(
        (
            line.strip()
            for line in lines
            if re.match(r"^add_files\b", line.strip())
            and "-tb" not in line
            and re.search(r"\.(?:c|cc|cpp|cxx)(?:\s|$)", line.strip())
        ),
        None,
    )
    if design is None:
        raise ValueError("Could not find baseline design-source add_files command.")
    auxiliaries = [
        line.strip()
        for line in lines
        if re.match(r"^add_files\b", line.strip())
        and (
            (include_testbench and "-tb" in line)
            or ("-tb" not in line and re.search(r"\.(?:h|hpp)(?:\s|$)", line.strip()))
        )
    ]
    if include_testbench and not any("-tb" in line for line in auxiliaries):
        raise ValueError("Could not find baseline testbench add_files command.")
    return (
        [set_top, open_solution, set_part, create_clock],
        _absolute_add_files(design, tcl_dir, candidate),
        [_absolute_add_files(line, tcl_dir) for line in auxiliaries],
        top_name,
    )


def _project_key(value: Path) -> str:
    return hashlib.sha256(str(value.resolve()).encode()).hexdigest()[:12]


def _terminate_process_group(process: subprocess.Popen[str], grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_vitis(tcl: Path, cwd: Path, log_path: Path, timeout_seconds: int) -> VitisResult:
    command = [find_vitis_run(), "--mode", "hls", "--tcl", str(tcl.resolve())]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started = time.monotonic()
    timed_out = False
    lines: list[str] = []
    stream_closed = False

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        while True:
            elapsed = time.monotonic() - started
            if process.poll() is None and elapsed >= timeout_seconds:
                timed_out = True
                message = f"\nTIMEOUT: Vitis exceeded {timeout_seconds} seconds; terminating process group.\n"
                print(message, end="", flush=True)
                log.write(message)
                lines.append(message)
                _terminate_process_group(process)

            try:
                item = output_queue.get(timeout=0.2)
            except queue.Empty:
                item = ""
            if item is None:
                stream_closed = True
            elif item:
                print(item, end="", flush=True)
                log.write(item)
                lines.append(item)

            if process.poll() is not None and stream_closed and output_queue.empty():
                break

    reader.join(timeout=1.0)
    return VitisResult(
        returncode=process.returncode if process.returncode is not None else -1,
        output="".join(lines),
        timed_out=timed_out,
        timeout_seconds=timeout_seconds,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )


def _timeout(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get("timeouts", {}).get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"timeouts.{key} must be a positive integer")
    return value


def run_candidate_csim(config_path: Path, candidate_index: int = 1) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    output_dir = REPO_ROOT / config["output_dir"]
    candidate = output_dir / f"candidate_{candidate_index:03d}.cpp"
    static_report = output_dir / f"candidate_{candidate_index:03d}_static_validation.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate not found: {candidate}")
    if not static_report.is_file() or load_json(static_report).get("passed") is not True:
        raise RuntimeError("Static validation did not pass; refusing to run CSim.")
    baseline_tcl = REPO_ROOT / config["baseline"]["tcl"]
    parts, design, auxiliaries, _ = _tcl_parts(baseline_tcl, candidate, True)
    set_top, open_solution, set_part, create_clock = parts
    project_dir = TMP_ROOT / _project_key(config_path) / f"candidate_{candidate_index:03d}_csim"
    csim_dir = output_dir / f"candidate_{candidate_index:03d}_csim"
    csim_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(project_dir, ignore_errors=True)
    project_dir.parent.mkdir(parents=True, exist_ok=True)
    generated_tcl = csim_dir / "run_csim.tcl"
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
                "csim_design",
                "exit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = csim_dir / "vitis_csim.log"
    completed = _run_vitis(
        generated_tcl,
        baseline_tcl.parent,
        log_path,
        _timeout(config, "csim_seconds", DEFAULT_CSIM_TIMEOUT_SECONDS),
    )
    candidate_compiled = bool(
        re.search(rf"Compiling\s+.*{re.escape(candidate.name)}\b", completed.output, re.I)
    )
    report = {
        "candidate_index": candidate_index,
        "candidate_file": str(candidate.relative_to(REPO_ROOT)),
        "generated_tcl": str(generated_tcl.relative_to(REPO_ROOT)),
        "design_source_command": design,
        "auxiliary_source_commands": auxiliaries,
        "project_dir": str(project_dir),
        "project_storage": "temporary",
        "log_file": str(log_path.relative_to(REPO_ROOT)),
        "return_code": completed.returncode,
        "candidate_compiled": candidate_compiled,
        "timed_out": completed.timed_out,
        "timeout_seconds": completed.timeout_seconds,
        "elapsed_seconds": completed.elapsed_seconds,
        "passed": completed.returncode == 0 and candidate_compiled and not completed.timed_out,
        "synthesis_run": False,
        "baseline_modified": False,
    }
    write_json(output_dir / f"candidate_{candidate_index:03d}_csim_validation.json", report)
    return report


def _text(root: ET.Element, path: str) -> str | None:
    node = root.find(path)
    return node.text.strip() if node is not None and node.text else None


def _number(value: str | None) -> int | float | None:
    if value in {None, "", "NA", "N/A"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def parse_csynth_xml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    return {
        "clock_period_ns": _number(_text(root, ".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")),
        "latency_best_cycles": _number(_text(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency")),
        "latency_average_cycles": _number(_text(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency")),
        "latency_worst_cycles": _number(_text(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency")),
        "interval_min_cycles": _number(_text(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Interval-min")),
        "interval_max_cycles": _number(_text(root, ".//PerformanceEstimates/SummaryOfOverallLatency/Interval-max")),
        "resources_lut_used": _number(_text(root, ".//AreaEstimates/Resources/LUT")),
        "resources_ff_used": _number(_text(root, ".//AreaEstimates/Resources/FF")),
        "resources_dsp_used": _number(_text(root, ".//AreaEstimates/Resources/DSP")),
        "resources_bram_used": _number(_text(root, ".//AreaEstimates/Resources/BRAM_18K")),
    }


def run_candidate_synthesis(
    config_path: Path,
    candidate_index: int = 1,
    extract_only: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    candidate = output_dir / f"candidate_{candidate_index:03d}.cpp"
    csim_report = output_dir / f"candidate_{candidate_index:03d}_csim_validation.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate not found: {candidate}")
    if not csim_report.is_file() or load_json(csim_report).get("passed") is not True:
        raise RuntimeError("Candidate CSim did not pass; refusing to run synthesis.")
    baseline_tcl = REPO_ROOT / config["baseline"]["tcl"]
    parts, design, headers, top_name = _tcl_parts(baseline_tcl, candidate, False)
    set_top, open_solution, set_part, create_clock = parts
    project_dir = TMP_ROOT / _project_key(output_dir) / f"candidate_{candidate_index:03d}_synthesis"
    synthesis_dir = output_dir / f"candidate_{candidate_index:03d}_synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)
    if not extract_only:
        shutil.rmtree(project_dir, ignore_errors=True)
    project_dir.parent.mkdir(parents=True, exist_ok=True)
    generated_tcl = synthesis_dir / "run_synthesis.tcl"
    generated_tcl.write_text(
        "\n".join(
            [
                f'open_project -reset "{project_dir.resolve().as_posix()}"',
                set_top,
                design,
                *headers,
                open_solution,
                set_part,
                create_clock,
                "csynth_design",
                "exit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = synthesis_dir / "vitis_synthesis.log"
    return_code: int | None = None
    timed_out = False
    elapsed_seconds: float | None = None
    timeout_seconds = _timeout(config, "synthesis_seconds", DEFAULT_SYNTHESIS_TIMEOUT_SECONDS)
    if not extract_only:
        completed = _run_vitis(generated_tcl, baseline_tcl.parent, log_path, timeout_seconds)
        return_code = completed.returncode
        timed_out = completed.timed_out
        elapsed_seconds = completed.elapsed_seconds

    report_dir = project_dir / "solution1/syn/report"
    hierarchy: dict[str, dict[str, Any]] = {}
    parse_error: str | None = None
    try:
        for xml_path in sorted(report_dir.glob("*_csynth.xml")):
            name = xml_path.name.removesuffix("_csynth.xml")
            hierarchy[name] = {
                "csynth_xml": str(xml_path),
                "metrics": parse_csynth_xml(xml_path),
            }
    except (ET.ParseError, OSError, ValueError) as error:
        parse_error = str(error)
    top_report: Path | None = report_dir / f"{top_name}_csynth.xml"
    if not top_report.is_file():
        top_report = None
    report = {
        "candidate_index": candidate_index,
        "candidate_file": str(candidate.relative_to(REPO_ROOT)),
        "generated_tcl": str(generated_tcl.relative_to(REPO_ROOT)),
        "design_source_command": design,
        "auxiliary_source_commands": headers,
        "temporary_project_dir": str(project_dir),
        "log_file": str(log_path.relative_to(REPO_ROOT)) if log_path.exists() else None,
        "top_function": top_name,
        "top_csynth_xml": str(top_report) if top_report else None,
        "return_code": return_code,
        "extract_only": extract_only,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": elapsed_seconds,
        "failure_class": "synthesis_timeout" if timed_out else ("synthesis_failed" if not extract_only and return_code != 0 else None),
        "passed": top_report is not None and (extract_only or (return_code == 0 and not timed_out)),
        "metrics": hierarchy.get(top_name, {}).get("metrics", {}),
        "hierarchical_reports": hierarchy,
        "parse_error": parse_error,
        "synthesis_run": not extract_only,
        "baseline_modified": False,
    }
    write_json(output_dir / f"candidate_{candidate_index:03d}_synthesis.json", report)
    return report


def ensure_baseline_synthesis(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path.resolve())
    baseline = config["baseline"]
    baseline_tcl = REPO_ROOT / baseline["tcl"]
    baseline_source = REPO_ROOT / baseline["source"]
    project_dir = REPO_ROOT / baseline["project_dir"]
    output_dir = REPO_ROOT / config["output_dir"]
    existing = sorted(project_dir.rglob("*_csynth.xml")) if project_dir.is_dir() else []
    if existing:
        return {"passed": True, "cached": True, "reports": len(existing), "project_dir": str(project_dir)}
    parts, design, auxiliaries, _ = _tcl_parts(baseline_tcl, baseline_source, True)
    set_top, open_solution, set_part, create_clock = parts
    run_dir = output_dir / "baseline_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    generated_tcl = run_dir / "run_baseline.tcl"
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
                "csim_design",
                "csynth_design",
                "exit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = run_dir / "vitis_baseline.log"
    completed = _run_vitis(
        generated_tcl,
        baseline_tcl.parent,
        log_path,
        _timeout(config, "baseline_seconds", DEFAULT_SYNTHESIS_TIMEOUT_SECONDS),
    )
    reports = sorted(project_dir.rglob("*_csynth.xml")) if project_dir.is_dir() else []
    return {
        "passed": completed.returncode == 0 and bool(reports) and not completed.timed_out,
        "cached": False,
        "reports": len(reports),
        "project_dir": str(project_dir),
        "return_code": completed.returncode,
        "timed_out": completed.timed_out,
        "timeout_seconds": completed.timeout_seconds,
        "elapsed_seconds": completed.elapsed_seconds,
        "log_file": str(log_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vitis validation and synthesis tools.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("csim", "synth"):
        child = subparsers.add_parser(action)
        child.add_argument("config", type=Path)
        child.add_argument("--candidate-index", type=int, default=1)
        if action == "synth":
            child.add_argument("--extract-only", action="store_true")
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("config", type=Path)
    args = parser.parse_args()
    if args.action == "csim":
        report = run_candidate_csim(args.config, args.candidate_index)
    elif args.action == "synth":
        report = run_candidate_synthesis(args.config, args.candidate_index, args.extract_only)
    else:
        report = ensure_baseline_synthesis(args.config)
    print(f"Overall: {'PASS' if report.get('passed') else 'FAIL'}")
    raise SystemExit(0 if report.get("passed") else 1)


if __name__ == "__main__":
    main()
