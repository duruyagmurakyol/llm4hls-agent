#!/usr/bin/env python3

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def read_config_value(config_path: Path, key: str) -> str | None:
    """Read a simple key=value entry from a Vitis HLS config file."""

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or line.startswith("["):
            continue

        config_key, separator, value = line.partition("=")

        if separator and config_key.strip() == key:
            return value.strip()

    return None


def classify_result(return_code: int, output: str) -> tuple[str, str]:
    """Classify the C simulation result."""

    output_lower = output.lower()

    if return_code == 0 and "csim done with 0 errors" in output_lower:
        return "passed", "none"

    compile_markers = [
        "error: expected",
        "error: use of undeclared identifier",
        "error: no matching function",
        "error: unknown type name",
        "error: fatal error",
        "compilation terminated",
    ]

    if any(marker in output_lower for marker in compile_markers):
        return "failed", "compile"

    functional_markers = [
        "fail",
        "returns nonzero value",
        "nonzero return value",
        "simulation failed",
    ]

    if any(marker in output_lower for marker in functional_markers):
        return "failed", "functional"

    return "failed", "unknown"


def extract_evidence(output: str) -> list[str]:
    """Extract useful failure lines from the Vitis output."""

    keywords = [
        "error:",
        "fail",
        "simulation failed",
        "csim failed",
        "returns nonzero",
        "nonzero return",
    ]

    evidence: list[str] = []

    for line in output.splitlines():
        line_lower = line.lower()

        if any(keyword in line_lower for keyword in keywords):
            cleaned_line = line.strip()

            if cleaned_line and cleaned_line not in evidence:
                evidence.append(cleaned_line)

    return evidence[:20]


def generate_prompt(
    benchmark_name: str,
    failure_class: str,
    evidence: list[str],
    source_code: str,
) -> str:
    """Generate an LLM repair prompt from the diagnosed failure."""

    evidence_text = (
        "\n".join(evidence)
        if evidence
        else "No specific error evidence was extracted."
    )

    return f"""You are repairing a Xilinx HLS C++ benchmark.

Benchmark: {benchmark_name}
Failure class: {failure_class}

Vitis HLS failure evidence:
{evidence_text}

Repair the implementation so that it passes C simulation.

Preserve:
- the function name and arguments
- the existing loop structure where possible
- all HLS pragmas
- compatibility with Xilinx Vitis HLS

Return only the corrected complete source file.

Current source:

{source_code}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and diagnose a Vitis HLS C simulation."
    )
    parser.add_argument(
        "benchmark",
        type=Path,
        help="Path to the benchmark directory containing task.cfg",
    )

    args = parser.parse_args()

    benchmark_dir = args.benchmark.expanduser().resolve()
    config_path = benchmark_dir / "task.cfg"

    if not benchmark_dir.is_dir():
        raise SystemExit(f"Benchmark directory not found: {benchmark_dir}")

    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")

    source_setting = read_config_value(config_path, "syn.file")

    if source_setting is None:
        raise SystemExit(f"syn.file is missing from config: {config_path}")

    source_path = benchmark_dir / source_setting

    if not source_path.is_file():
        raise SystemExit(f"Kernel source not found: {source_path}")

    repository_root = Path(__file__).resolve().parent.parent

    try:
        benchmark_name = benchmark_dir.relative_to(repository_root).as_posix()
    except ValueError:
        benchmark_name = benchmark_dir.name

    safe_name = benchmark_name.replace("/", "_").replace("\\", "_")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repository_root / "runs" / safe_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    work_dir = run_dir / "work"
    log_path = run_dir / "vitis_csim.log"
    result_path = run_dir / "result.json"
    prompt_path = run_dir / "repair_prompt.txt"

    command = [
        "vitis-run",
        "--mode",
        "hls",
        "--csim",
        "--config",
        str(config_path),
        "--work_dir",
        str(work_dir),
    ]

    print(f"Benchmark: {benchmark_name}")
    print(f"Source: {source_path}")
    print("Running C simulation...")

    try:
        process = subprocess.run(
            command,
            cwd=benchmark_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        raise SystemExit(
            "vitis-run was not found. Source the Vitis environment first."
        )

    output = process.stdout or ""
    log_path.write_text(output, encoding="utf-8")

    status, failure_class = classify_result(process.returncode, output)
    evidence = extract_evidence(output)

    result: dict[str, object] = {
        "benchmark": benchmark_name,
        "benchmark_path": str(benchmark_dir),
        "source": str(source_path),
        "timestamp": timestamp,
        "stage": "csim",
        "status": status,
        "failure_class": failure_class,
        "return_code": process.returncode,
        "evidence": evidence,
        "log": str(log_path.relative_to(repository_root)),
    }

    if status == "failed":
        source_code = source_path.read_text(encoding="utf-8")

        prompt = generate_prompt(
            benchmark_name=benchmark_name,
            failure_class=failure_class,
            evidence=evidence,
            source_code=source_code,
        )

        prompt_path.write_text(prompt, encoding="utf-8")
        result["repair_prompt"] = str(
            prompt_path.relative_to(repository_root)
        )

    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"Result: {status.upper()}")
    print(f"Failure class: {failure_class}")

    if evidence:
        print("Evidence:")

        for line in evidence:
            print(f"  {line}")

    if status == "failed":
        print(f"Generated prompt: {prompt_path}")

    print(f"Log: {log_path}")
    print(f"Result record: {result_path}")


if __name__ == "__main__":
    main()