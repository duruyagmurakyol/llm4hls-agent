#!/usr/bin/env python3

import argparse
import json
import shutil
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
    """Classify a Vitis HLS C-simulation result."""

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
    """Extract useful failure lines from tool output."""

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
    """Generate an LLM repair prompt."""

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
Ensure the source is syntactically complete.
Do not use Markdown code fences or explanatory text.

Current source:

{source_code}
"""


def run_command(
    command: list[str],
    cwd: Path,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command while capturing combined output."""

    try:
        return subprocess.run(
            command,
            cwd=cwd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as error:
        raise SystemExit(f"Command not found: {command[0]}") from error


def run_vitis_csim(
    benchmark_dir: Path,
    config_path: Path,
    work_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Run Vitis HLS C simulation."""

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

    return run_command(command, cwd=benchmark_dir)


def run_codex(
    prompt_text: str,
    output_path: Path,
    log_path: Path,
    repository_root: Path,
    model: str,
) -> subprocess.CompletedProcess[str]:
    """Generate a proposed repair using Codex CLI."""

    command = [
        "codex",
        "exec",
        "-m",
        model,
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(output_path),
        "-",
    ]

    process = run_command(
        command,
        cwd=repository_root,
        input_text=prompt_text,
    )

    log_path.write_text(process.stdout or "", encoding="utf-8")
    return process


def copy_validation_files(
    benchmark_dir: Path,
    config_path: Path,
    source_path: Path,
    candidate_path: Path,
    validation_dir: Path,
) -> tuple[Path, Path]:
    """Create an isolated validation copy of the benchmark."""

    validation_dir.mkdir(parents=True, exist_ok=True)

    source_relative = source_path.relative_to(benchmark_dir)
    validation_source = validation_dir / source_relative
    validation_source.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(candidate_path, validation_source)

    tb_setting = read_config_value(config_path, "tb.file")

    if tb_setting is None:
        raise SystemExit(f"tb.file is missing from config: {config_path}")

    tb_path = benchmark_dir / tb_setting

    if not tb_path.is_file():
        raise SystemExit(f"Testbench file not found: {tb_path}")

    tb_relative = tb_path.relative_to(benchmark_dir)
    validation_tb = validation_dir / tb_relative
    validation_tb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tb_path, validation_tb)

    for header in benchmark_dir.rglob("*.h"):
        relative_header = header.relative_to(benchmark_dir)
        destination = validation_dir / relative_header
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(header, destination)

    validation_config = validation_dir / "task.cfg"
    shutil.copy2(config_path, validation_config)

    return validation_config, validation_source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run, diagnose, repair and validate a Vitis HLS benchmark."
    )

    parser.add_argument(
        "benchmark",
        type=Path,
        help="Benchmark directory containing task.cfg",
    )

    parser.add_argument(
        "--generate-repair",
        action="store_true",
        help="Use Codex to generate a candidate repair after failure",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated candidate using Vitis C simulation",
    )

    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="Codex model to use",
    )

    args = parser.parse_args()

    if args.validate and not args.generate_repair:
        raise SystemExit("--validate requires --generate-repair")

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

    original_work_dir = run_dir / "original_work"
    original_log_path = run_dir / "vitis_csim.log"
    result_path = run_dir / "result.json"
    prompt_path = run_dir / "repair_prompt.txt"
    candidate_path = run_dir / "codex_repair.cpp"
    codex_log_path = run_dir / "codex_exec.log"

    print(f"Benchmark: {benchmark_name}")
    print(f"Source: {source_path}")
    print("Running original C simulation...")

    original_process = run_vitis_csim(
        benchmark_dir=benchmark_dir,
        config_path=config_path,
        work_dir=original_work_dir,
    )

    original_output = original_process.stdout or ""
    original_log_path.write_text(original_output, encoding="utf-8")

    status, failure_class = classify_result(
        original_process.returncode,
        original_output,
    )

    evidence = extract_evidence(original_output)

    result: dict[str, object] = {
        "benchmark": benchmark_name,
        "benchmark_path": str(benchmark_dir),
        "source": str(source_path),
        "timestamp": timestamp,
        "stage": "csim",
        "status": status,
        "failure_class": failure_class,
        "return_code": original_process.returncode,
        "evidence": evidence,
        "log": str(original_log_path.relative_to(repository_root)),
        "repair_requested": args.generate_repair,
        "validation_requested": args.validate,
    }

    print(f"Original result: {status.upper()}")
    print(f"Failure class: {failure_class}")

    if evidence:
        print("Evidence:")

        for line in evidence:
            print(f"  {line}")

    if status == "passed":
        result["repair_status"] = "not_required"

        result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print("Benchmark already passes; no repair generated.")
        print(f"Result record: {result_path}")
        return

    source_code = source_path.read_text(encoding="utf-8")

    prompt = generate_prompt(
        benchmark_name=benchmark_name,
        failure_class=failure_class,
        evidence=evidence,
        source_code=source_code,
    )

    prompt_path.write_text(prompt, encoding="utf-8")
    result["repair_prompt"] = str(prompt_path.relative_to(repository_root))

    print(f"Generated prompt: {prompt_path}")

    if not args.generate_repair:
        result["repair_status"] = "prompt_generated"

        result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print(f"Result record: {result_path}")
        return

    print(f"Requesting repair from Codex using {args.model}...")

    codex_process = run_codex(
        prompt_text=prompt,
        output_path=candidate_path,
        log_path=codex_log_path,
        repository_root=repository_root,
        model=args.model,
    )

    result["codex_model"] = args.model
    result["codex_return_code"] = codex_process.returncode
    result["codex_log"] = str(codex_log_path.relative_to(repository_root))

    if codex_process.returncode != 0:
        result["repair_status"] = "codex_failed"

        result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        raise SystemExit(
            f"Codex failed. See: {codex_log_path}"
        )

    if not candidate_path.is_file():
        result["repair_status"] = "candidate_missing"

        result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        raise SystemExit("Codex did not produce a candidate source file.")

    candidate_text = candidate_path.read_text(encoding="utf-8").strip()

    if not candidate_text:
        result["repair_status"] = "candidate_empty"

        result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        raise SystemExit("Codex produced an empty candidate.")

    result["repair_status"] = "candidate_generated"
    result["candidate"] = str(candidate_path.relative_to(repository_root))

    print(f"Candidate repair: {candidate_path}")

    if not args.validate:
        result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print(f"Result record: {result_path}")
        return

    print("Validating candidate in an isolated benchmark copy...")

    validation_dir = run_dir / "validation"

    validation_config, _ = copy_validation_files(
        benchmark_dir=benchmark_dir,
        config_path=config_path,
        source_path=source_path,
        candidate_path=candidate_path,
        validation_dir=validation_dir,
    )

    validation_work_dir = run_dir / "validation_work"
    validation_log_path = run_dir / "validation_csim.log"

    validation_process = run_vitis_csim(
        benchmark_dir=validation_dir,
        config_path=validation_config,
        work_dir=validation_work_dir,
    )

    validation_output = validation_process.stdout or ""
    validation_log_path.write_text(validation_output, encoding="utf-8")

    validation_status, validation_failure_class = classify_result(
        validation_process.returncode,
        validation_output,
    )

    validation_evidence = extract_evidence(validation_output)

    result["validation"] = {
        "status": validation_status,
        "failure_class": validation_failure_class,
        "return_code": validation_process.returncode,
        "evidence": validation_evidence,
        "log": str(validation_log_path.relative_to(repository_root)),
    }

    if validation_status == "passed":
        result["repair_status"] = "validated"
        print("Validation result: PASSED")
    else:
        result["repair_status"] = "validation_failed"
        print("Validation result: FAILED")

        if validation_evidence:
            print("Validation evidence:")

            for line in validation_evidence:
                print(f"  {line}")

    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"Result record: {result_path}")


if __name__ == "__main__":
    main()