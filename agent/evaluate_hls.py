#!/usr/bin/env python3

import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def parse_csynth_report(project_dir: Path) -> dict[str, Any]:
    """
    Find and parse the Vitis HLS synthesis report.

    Prefers XML because it is more reliable than parsing formatted text.
    """

    xml_reports = list(project_dir.rglob("*csynth.xml"))

    if not xml_reports:
        return {}

    report_path = xml_reports[0]

    try:
        root = ET.parse(report_path).getroot()
    except ET.ParseError:
        return {}

    def find_text(*paths: str) -> str | None:
        for path in paths:
            element = root.find(path)
            if element is not None and element.text is not None:
                return element.text.strip()
        return None

    def to_int(value: str | None) -> int | None:
        if value is None or value in {"", "N/A", "-"}:
            return None

        try:
            return int(float(value))
        except ValueError:
            return None

    def to_float(value: str | None) -> float | None:
        if value is None or value in {"", "N/A", "-"}:
            return None

        try:
            return float(value)
        except ValueError:
            return None

    latency_best = find_text(
        ".//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency",
        ".//SummaryOfOverallLatency/Best-caseLatency",
    )

    latency_average = find_text(
        ".//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency",
        ".//SummaryOfOverallLatency/Average-caseLatency",
    )

    latency_worst = find_text(
        ".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency",
        ".//SummaryOfOverallLatency/Worst-caseLatency",
    )

    estimated_clock = find_text(
        ".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod",
        ".//SummaryOfTimingAnalysis/EstimatedClockPeriod",
    )

    target_clock = find_text(
        ".//PerformanceEstimates/SummaryOfTimingAnalysis/TargetClockPeriod",
        ".//SummaryOfTimingAnalysis/TargetClockPeriod",
    )

    lut = find_text(
        ".//AreaEstimates/Resources/LUT",
        ".//Resources/LUT",
    )

    ff = find_text(
        ".//AreaEstimates/Resources/FF",
        ".//Resources/FF",
    )

    dsp = find_text(
        ".//AreaEstimates/Resources/DSP",
        ".//Resources/DSP",
    )

    bram = find_text(
        ".//AreaEstimates/Resources/BRAM_18K",
        ".//Resources/BRAM_18K",
    )

    metrics = {
        "latency_best_cycles": to_int(latency_best),
        "latency_average_cycles": to_int(latency_average),
        "latency_worst_cycles": to_int(latency_worst),
        "estimated_clock_period_ns": to_float(estimated_clock),
        "target_clock_period_ns": to_float(target_clock),
        "resources_lut_used": to_int(lut),
        "resources_ff_used": to_int(ff),
        "resources_dsp_used": to_int(dsp),
        "resources_bram_used": to_int(bram),
        "report_path": str(report_path),
    }

    return {
        key: value
        for key, value in metrics.items()
        if value is not None
    }


def evaluate_candidate(
    candidate_path: str,
    benchmark_dir: str = "benchmarks/hls_eval/atax",
    run_dir: str = "runs/atax_agent",
) -> dict[str, Any]:
    """
    Evaluate one ATAX candidate with Vitis HLS.

    The candidate replaces benchmarks/hls_eval/atax/src/atax.cpp.
    A fresh Vitis project is created inside run_dir.
    """

    candidate = Path(candidate_path).resolve()
    benchmark = Path(benchmark_dir).resolve()
    output = Path(run_dir).resolve()

    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate not found: {candidate}")

    source_dir = benchmark / "src"
    testbench_dir = benchmark / "testbench"

    header = source_dir / "atax.h"
    testbench = testbench_dir / "atax_tb.cpp"

    for required_file in (header, testbench):
        if not required_file.is_file():
            raise FileNotFoundError(f"Required benchmark file missing: {required_file}")

    if output.exists():
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)

    work_src = output / "src"
    work_tb = output / "testbench"

    work_src.mkdir(exist_ok=True)
    work_tb.mkdir(exist_ok=True)

    shutil.copy2(candidate, work_src / "atax.cpp")
    shutil.copy2(header, work_src / "atax.h")
    shutil.copy2(testbench, work_tb / "atax_tb.cpp")

    tcl_path = output / "run_hls.tcl"

    tcl_path.write_text(
        f"""
open_project -reset {output / "project"}
set_top kernel_atax
add_files {work_src / "atax.cpp"}
add_files {work_src / "atax.h"}
add_files -tb {work_tb / "atax_tb.cpp"} -cflags "-I{work_src}"
open_solution -reset solution1
set_part {{xczu3eg-sfvc784-2-e}}
create_clock -period 10
csim_design
csynth_design
exit
""".strip()
        + "\n",
        encoding="utf-8",
    )

    log_path = output / "vitis.log"

    command = [
        "vitis-run",
        "--mode",
        "hls",
        "--tcl",
        str(tcl_path),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=output,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
            check=False,
        )

        log_path.write_text(completed.stdout, encoding="utf-8")

    except subprocess.TimeoutExpired as error:
        log_text = error.stdout or ""
        if isinstance(log_text, bytes):
            log_text = log_text.decode(errors="replace")

        log_path.write_text(log_text, encoding="utf-8")

        return {
            "csim_pass": False,
            "synth_pass": False,
            "timeout": True,
            "return_code": None,
            "metrics": {},
            "log_path": str(log_path),
        }

    log_text = completed.stdout

    csim_pass = (
        "CSim done with 0 errors" in log_text
        or "CSIM finished successfully" in log_text
        or "C simulation finished successfully" in log_text
    )

    synth_pass = (
        completed.returncode == 0
        and (
            "Finished Command csynth_design" in log_text
            or "Synthesis completed successfully" in log_text
        )
    )

    project_dir = output / "project"

    metrics = (
        parse_csynth_report(project_dir)
        if synth_pass
        else {}
    )

    result: dict[str, Any] = {
        "csim_pass": csim_pass,
        "synth_pass": synth_pass,
        "timeout": False,
        "return_code": completed.returncode,
        "metrics": metrics,
        "log_path": str(log_path),
    }

    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
