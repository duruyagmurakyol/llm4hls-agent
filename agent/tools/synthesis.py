"""Portable Vitis HLS CSim, synthesis and report extraction tools."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import re
import shlex
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

from agent.config import TaskManifest
from agent.tools.command_runner import CommandResult, run_command
from agent.tools.reports import load_json, write_json
from agent.tools.validation import classify_failure, extract_evidence

REPO_ROOT = Path(__file__).resolve().parents[2]
TMP_ROOT = Path(tempfile.gettempdir()) / "llm4hls-agent"
DEFAULT_CSIM_TIMEOUT_SECONDS = 300
DEFAULT_SYNTHESIS_TIMEOUT_SECONDS = 600


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


def _with_include_dir(options: list[str], include_dir: Path) -> list[str]:
    """Ensure an add_files option list contains the supplied include directory."""
    result = list(options)
    include_flag = f"-I{include_dir.resolve().as_posix()}"

    for index, token in enumerate(result[:-1]):
        if token != "-cflags":
            continue

        flags = shlex.split(result[index + 1])
        if include_flag not in flags:
            flags.append(include_flag)
        result[index + 1] = " ".join(flags)
        return result

    result.extend(["-cflags", include_flag])
    return result


def _absolute_add_files(command: str, tcl_dir: Path, replacement: Path | None = None) -> str:
    is_tb, options, source_text = _parse_add_files(command)

    original_source = Path(source_text)
    if not original_source.is_absolute():
        original_source = (tcl_dir / original_source).resolve()

    source = replacement.resolve() if replacement else original_source
    options = _absolute_options(options, tcl_dir)

    # A generated candidate normally lives outside the benchmark's original
    # source directory. Preserve that directory on the compiler include path
    # so includes such as #include "vector_add.h" continue to resolve.
    if replacement is not None and original_source.suffix.lower() in {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
    }:
        options = _with_include_dir(options, original_source.parent)

    tokens = ["add_files"]
    if is_tb and "-tb" not in options:
        tokens.append("-tb")
    tokens.extend(options)
    tokens.append(source.as_posix())
    return " ".join(shlex.quote(token) for token in tokens)


def _cfg_parts(
    cfg_path: Path,
    candidate: Path,
    include_testbench: bool,
) -> tuple[list[str], str, list[str], str]:
    parser = configparser.ConfigParser()
    parser.read(cfg_path, encoding="utf-8")

    if "hls" not in parser:
        raise ValueError(f"Missing [hls] section in {cfg_path}")

    hls = parser["hls"]
    required = ("syn.top", "syn.file", "part", "clock")
    missing = [key for key in required if not hls.get(key, "").strip()]
    if missing:
        raise ValueError(f"Missing required task.cfg fields: {', '.join(missing)}")

    cfg_dir = cfg_path.parent.resolve()
    top_name = hls["syn.top"].strip()
    source = (cfg_dir / hls["syn.file"].strip()).resolve()
    syn_cflags = hls.get("syn.cflags", "").strip()
    tb_cflags = hls.get("tb.cflags", "").strip()

    def absolute_cflags(value: str) -> str:
        flags = []
        for flag in shlex.split(value):
            if flag.startswith("-I") and len(flag) > 2:
                include = Path(flag[2:])
                if not include.is_absolute():
                    include = (cfg_dir / include).resolve()
                flag = f"-I{include.as_posix()}"
            flags.append(flag)
        return " ".join(flags)

    design_options: list[str] = []
    absolute_syn_cflags = absolute_cflags(syn_cflags)
    if absolute_syn_cflags:
        design_options.extend(["-cflags", absolute_syn_cflags])

    # Candidates are emitted outside the original source directory. Always
    # preserve that directory as an include path for local benchmark headers.
    design_options = _with_include_dir(design_options, source.parent)

    design_tokens = ["add_files", *design_options, candidate.resolve().as_posix()]
    design = " ".join(shlex.quote(token) for token in design_tokens)

    auxiliaries: list[str] = []
    for header in sorted(source.parent.glob("*.h*")):
        auxiliaries.append(f"add_files {shlex.quote(header.resolve().as_posix())}")

    if include_testbench:
        tb_value = hls.get("tb.file", "").strip()
        if not tb_value:
            raise ValueError("task.cfg is missing tb.file")
        testbench = (cfg_dir / tb_value).resolve()
        tb_tokens = ["add_files", "-tb"]
        absolute_tb_cflags = absolute_cflags(tb_cflags)
        if absolute_tb_cflags:
            tb_tokens.extend(["-cflags", absolute_tb_cflags])
        tb_tokens.append(testbench.as_posix())
        auxiliaries.append(" ".join(shlex.quote(token) for token in tb_tokens))

    clock_match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*ns\s*",
        hls["clock"],
        re.IGNORECASE,
    )
    if not clock_match:
        raise ValueError(f"Unsupported task.cfg clock value: {hls['clock']}")

    parts = [
        f"set_top {top_name}",
        "open_solution -reset solution1",
        f"set_part {hls['part'].strip()}",
        f"create_clock -period {clock_match.group(1)} -name default",
    ]
    return parts, design, auxiliaries, top_name


def _tcl_parts(
    baseline_tcl: Path,
    candidate: Path,
    include_testbench: bool,
) -> tuple[list[str], str, list[str], str]:
    if baseline_tcl.suffix.lower() == ".cfg":
        return _cfg_parts(baseline_tcl, candidate, include_testbench)

    lines = baseline_tcl.read_text(encoding="utf-8").splitlines()
    tcl_dir = baseline_tcl.parent.resolve()
    set_top = _first(lines, r"^set_top\b", "set_top")
    top_name = set_top.split()[-1].strip("{}\"")
    open_solution = _first(lines, r"^open_solution(?:\s+-reset)?\b", "open_solution")
    set_part = _first(lines, r"^set_part\b", "set_part")
    create_clock = _first(lines, r"^create_clock\b", "create_clock")
    design = None

    for line in lines:
        command = line.strip()

        if not re.match(r"^add_files\b", command):
            continue

        try:
            is_testbench, _, source_path = _parse_add_files(command)
        except ValueError:
            continue

        if (
            not is_testbench
            and re.search(r"\.(?:c|cc|cpp|cxx)$", source_path, re.IGNORECASE)
        ):
            design = command
            break
    if design is None:
        raise ValueError("Could not find baseline design-source add_files command.")
    auxiliaries: list[str] = []

    for line in lines:
        command = line.strip()

        if not re.match(r"^add_files\b", command):
            continue

        try:
            is_testbench, _, source_path = _parse_add_files(command)
        except ValueError:
            continue

        suffix = Path(source_path).suffix.lower()

        if include_testbench and is_testbench:
            auxiliaries.append(command)
        elif not is_testbench and suffix in {".h", ".hpp"}:
            auxiliaries.append(command)
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


def _candidate_hash(candidate: Path) -> str:
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _run_vitis(
    tcl: Path,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> CommandResult:
    result = run_command(
        [find_vitis_run(), "--mode", "hls", "--tcl", tcl.resolve()],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.output, encoding="utf-8")
    return result


def _timeout(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get("timeouts", {}).get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"timeouts.{key} must be a positive integer")
    return value


def run_csim(task: TaskManifest, candidate: Path) -> dict[str, Any]:
    """Run CSim for one candidate using an authoritative task manifest."""
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate not found: {candidate}")

    build_files = task.data["artifacts"].get("build_files", [])
    if not build_files:
        raise ValueError("Task manifest must define artifacts.build_files")

    build_file = _resolve(build_files[0])
    parts, design, auxiliaries, _ = _tcl_parts(build_file, candidate, True)
    set_top, open_solution, set_part, create_clock = parts
    digest = _candidate_hash(candidate)
    output_dir = _resolve(task.output_dir)
    run_dir = output_dir / "csim" / digest[:12]
    project_dir = TMP_ROOT / digest[:12] / "csim"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(project_dir, ignore_errors=True)
    project_dir.parent.mkdir(parents=True, exist_ok=True)

    generated_tcl = run_dir / "run_csim.tcl"
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

    log_path = run_dir / "vitis_csim.log"
    completed = _run_vitis(
        generated_tcl,
        build_file.parent,
        log_path,
        _timeout(task.data, "csim_seconds", DEFAULT_CSIM_TIMEOUT_SECONDS),
    )
    candidate_compiled = bool(
        re.search(rf"Compiling\s+.*{re.escape(candidate.name)}\b", completed.output, re.I)
    )
    passed = completed.passed and candidate_compiled
    report = {
        "passed": passed,
        "timed_out": completed.timed_out,
        "return_code": completed.return_code,
        "failure_class": (
            "none"
            if passed
            else classify_failure(
                completed.output,
                stage="csim",
                timed_out=completed.timed_out,
            )
        ),
        "evidence": [] if passed else extract_evidence(completed.output),
        "command": list(completed.command),
        "duration_seconds": completed.elapsed_seconds,
        "timeout_seconds": completed.timeout_seconds,
        "log_path": _display_path(log_path),
        "candidate_hash": digest,
        "candidate_file": _display_path(candidate),
        "candidate_compiled": candidate_compiled,
        "generated_tcl": _display_path(generated_tcl),
        "project_dir": str(project_dir),
    }
    write_json(run_dir / "result.json", report)
    return report


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
    passed = completed.passed and candidate_compiled
    report = {
        "candidate_index": candidate_index,
        "candidate_file": str(candidate.relative_to(REPO_ROOT)),
        "candidate_hash": _candidate_hash(candidate),
        "generated_tcl": str(generated_tcl.relative_to(REPO_ROOT)),
        "design_source_command": design,
        "auxiliary_source_commands": auxiliaries,
        "project_dir": str(project_dir),
        "project_storage": "temporary",
        "log_file": str(log_path.relative_to(REPO_ROOT)),
        "command": list(completed.command),
        "return_code": completed.return_code,
        "candidate_compiled": candidate_compiled,
        "timed_out": completed.timed_out,
        "timeout_seconds": completed.timeout_seconds,
        "elapsed_seconds": completed.elapsed_seconds,
        "failure_class": (
            "none"
            if passed
            else classify_failure(
                completed.output,
                stage="csim",
                timed_out=completed.timed_out,
            )
        ),
        "evidence": [] if passed else extract_evidence(completed.output),
        "passed": passed,
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


def _read_synthesis_reports(
    project_dir: Path,
    top_name: str,
) -> tuple[dict[str, dict[str, Any]], Path | None, str | None]:
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
    top_report = report_dir / f"{top_name}_csynth.xml"
    return hierarchy, top_report if top_report.is_file() else None, parse_error


def _synthesis_failure(
    *,
    output: str,
    timed_out: bool,
    parse_error: str | None,
    process_passed: bool,
    top_report: Path | None,
) -> tuple[str, list[str]]:
    if timed_out:
        return (
            classify_failure(output, stage="synthesis", timed_out=True),
            extract_evidence(output),
        )
    if parse_error:
        return "report_parse", [parse_error]
    if process_passed and top_report is None:
        return "tool_report_missing", ["The expected top-level synthesis report was not generated."]
    return classify_failure(output, stage="synthesis"), extract_evidence(output)


def run_synthesis(task: TaskManifest, candidate: Path) -> dict[str, Any]:
    """Synthesise one candidate using an authoritative task manifest."""
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate not found: {candidate}")

    build_files = task.data["artifacts"].get("build_files", [])
    if not build_files:
        raise ValueError("Task manifest must define artifacts.build_files")

    build_file = _resolve(build_files[0])
    parts, design, headers, top_name = _tcl_parts(build_file, candidate, False)
    set_top, open_solution, set_part, create_clock = parts
    digest = _candidate_hash(candidate)
    output_dir = _resolve(task.output_dir)
    run_dir = output_dir / "synthesis" / digest[:12]
    project_dir = TMP_ROOT / digest[:12] / "synthesis"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(project_dir, ignore_errors=True)
    project_dir.parent.mkdir(parents=True, exist_ok=True)

    generated_tcl = run_dir / "run_synthesis.tcl"
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

    log_path = run_dir / "vitis_synthesis.log"
    completed = _run_vitis(
        generated_tcl,
        build_file.parent,
        log_path,
        _timeout(task.data, "synthesis_seconds", DEFAULT_SYNTHESIS_TIMEOUT_SECONDS),
    )
    hierarchy, top_report, parse_error = _read_synthesis_reports(project_dir, top_name)
    passed = completed.passed and top_report is not None and parse_error is None

    if passed:
        failure_class = "none"
        evidence: list[str] = []
    else:
        failure_class, evidence = _synthesis_failure(
            output=completed.output,
            timed_out=completed.timed_out,
            parse_error=parse_error,
            process_passed=completed.passed,
            top_report=top_report,
        )
        if failure_class == "tool_report_missing":
            evidence = [f"Missing top synthesis report for {top_name}"]

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
        "top_csynth_xml": str(top_report) if top_report else None,
        "metrics": hierarchy.get(top_name, {}).get("metrics", {}),
        "hierarchical_reports": hierarchy,
        "parse_error": parse_error,
        "synthesis_run": True,
        "baseline_modified": False,
    }
    write_json(run_dir / "result.json", report)
    return report


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
    output = ""
    timeout_seconds = _timeout(config, "synthesis_seconds", DEFAULT_SYNTHESIS_TIMEOUT_SECONDS)
    if not extract_only:
        completed = _run_vitis(generated_tcl, baseline_tcl.parent, log_path, timeout_seconds)
        return_code = completed.return_code
        timed_out = completed.timed_out
        elapsed_seconds = completed.elapsed_seconds
        output = completed.output

    hierarchy, top_report, parse_error = _read_synthesis_reports(project_dir, top_name)
    passed = top_report is not None and parse_error is None and (
        extract_only or (return_code == 0 and not timed_out)
    )
    if passed:
        failure_class = "none"
        evidence: list[str] = []
    elif extract_only and parse_error:
        failure_class = "report_parse"
        evidence = [parse_error]
    elif extract_only:
        failure_class = "tool_report_missing"
        evidence = [f"Missing top synthesis report for {top_name}"]
    else:
        failure_class, evidence = _synthesis_failure(
            output=output,
            timed_out=timed_out,
            parse_error=parse_error,
            process_passed=return_code == 0,
            top_report=top_report,
        )
        if failure_class == "tool_report_missing":
            evidence = [f"Missing top synthesis report for {top_name}"]

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
        "failure_class": failure_class,
        "evidence": evidence,
        "passed": passed,
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
        "passed": completed.return_code == 0 and bool(reports) and not completed.timed_out,
        "cached": False,
        "reports": len(reports),
        "project_dir": str(project_dir),
        "return_code": completed.return_code,
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
