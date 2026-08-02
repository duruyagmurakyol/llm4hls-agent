#!/usr/bin/env python3

"""Run a bounded iterative HLS repair loop with validation feedback."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent.siliconflow import complete, extract_source_response  # noqa: E402


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(config: dict[str, Any], workspace: Path) -> tuple[int, str]:
    host = config["host_validation"]
    compiled = run([str(x) for x in host["command"]], workspace)
    output = compiled.stdout or ""
    if compiled.returncode != 0:
        return compiled.returncode, output
    executed = run([str(x) for x in host["run_command"]], workspace)
    return executed.returncode, output + (executed.stdout or "")


def classify_failure(output: str) -> str:
    lower = output.lower()
    if "undefined reference" in lower or "linker" in lower:
        return "interface_or_link"
    if "error:" in lower or ("expected" in lower and "before" in lower):
        return "compile"
    if "fail index=" in lower or ("expected=" in lower and "actual=" in lower):
        return "functional"
    return "unknown"


def concise_evidence(output: str, limit: int = 1800) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    selected = [
        line
        for line in lines
        if any(
            token in line.lower()
            for token in ("error", "undefined", "fail", "expected", "actual")
        )
    ]
    text = "\n".join(selected[-16:] or lines[-16:])
    return text[-limit:]


def make_prompt(
    config: dict[str, Any],
    workspace: Path,
    iteration: int,
    evidence: str,
    category: str,
) -> tuple[str, str]:
    editable = str(config["editable_files"][0])
    source = (workspace / editable).read_text(encoding="utf-8")
    contexts: list[str] = []
    for name in config.get("context_files", config["protected_files"]):
        contexts.append(
            f"FILE: {name}\n```\n{(workspace / name).read_text(encoding='utf-8')}\n```"
        )

    system = (
        "You repair AMD/Xilinx HLS C++ code. Return only the complete repaired contents "
        "of the editable source file. Do not use Markdown fences, explanations, JSON, or patches. "
        "Preserve the declared top-function interface and HLS pragmas. Make the smallest complete "
        "repair that addresses all visible failures."
    )
    feedback_heading = (
        "Initial validation evidence" if iteration == 1 else "Feedback from the previous repair attempt"
    )
    user = (
        f"Repair iteration: {iteration}\n"
        f"Failure class: {category}\n"
        f"{feedback_heading}:\n{evidence}\n\n"
        f"EDITABLE FILE: {editable}\n```\n{source}\n```\n\n"
        + "\n\n".join(contexts)
        + "\n\nReturn only the full repaired editable file."
    )
    return system, user


def changed_line_count(before: str, after: str) -> tuple[int, str]:
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before/source.cpp",
            tofile="after/source.cpp",
        )
    )
    count = sum(
        1
        for line in diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )
    return count, diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("repair_mode") != "iterative_direct_api":
        raise SystemExit("Config must use repair_mode=iterative_direct_api")

    max_iterations = args.max_iterations or int(config.get("max_iterations", 3))
    if max_iterations < 1 or max_iterations > 10:
        raise SystemExit("max_iterations must be between 1 and 10")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = str(config["experiment_id"])
    run_dir = root / "runs" / "experiments" / experiment_id / timestamp
    workspace = run_dir / "workspace"
    run_dir.mkdir(parents=True)
    benchmark_source = root / config["benchmark_source"]
    if not benchmark_source.is_dir():
        raise SystemExit(f"Benchmark source not found: {benchmark_source}")
    shutil.copytree(benchmark_source, workspace)
    metadata = workspace / "fault.txt"
    if metadata.exists():
        metadata.unlink()

    editable = [str(x) for x in config["editable_files"]]
    protected = [str(x) for x in config["protected_files"]]
    protected_before = {name: sha256(workspace / name) for name in protected}
    initial_source = (workspace / editable[0]).read_text(encoding="utf-8")
    (run_dir / "before.cpp").write_text(initial_source, encoding="utf-8")

    validation_code, validation_output = validate(config, workspace)
    (run_dir / "validation_initial.log").write_text(validation_output, encoding="utf-8")
    initial_failure_class = classify_failure(validation_output)

    iterations: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_latency = 0.0
    success_iteration: int | None = None

    for iteration in range(1, max_iterations + 1):
        if validation_code == 0:
            break

        iteration_dir = run_dir / "iterations" / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True)
        before_source = (workspace / editable[0]).read_text(encoding="utf-8")
        evidence = concise_evidence(validation_output)
        category = classify_failure(validation_output)
        system_prompt, user_prompt = make_prompt(
            config, workspace, iteration, evidence, category
        )
        (iteration_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
        (iteration_dir / "prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")
        (iteration_dir / "feedback.log").write_text(validation_output, encoding="utf-8")
        (iteration_dir / "before.cpp").write_text(before_source, encoding="utf-8")

        thinking_budget_value = config.get("thinking_budget")
        response = complete(
            model=str(config["model"]),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_output_tokens", 2048)),
            timeout_seconds=int(config.get("api_timeout_seconds", 120)),
            thinking_budget=(
                int(thinking_budget_value) if thinking_budget_value is not None else None
            ),
        )
        (iteration_dir / "raw_response.txt").write_text(response.content, encoding="utf-8")
        (iteration_dir / "api_response.json").write_text(
            json.dumps(response.raw_response, indent=2) + "\n", encoding="utf-8"
        )

        try:
            candidate = extract_source_response(response.content)
        except ValueError as exc:
            validation_code = 1
            validation_output = f"Response parsing error: {exc}\n"
            (iteration_dir / "validation.log").write_text(validation_output, encoding="utf-8")
            candidate = before_source
        else:
            (iteration_dir / "candidate.cpp").write_text(candidate, encoding="utf-8")
            (workspace / editable[0]).write_text(candidate, encoding="utf-8")
            validation_code, validation_output = validate(config, workspace)
            (iteration_dir / "validation.log").write_text(validation_output, encoding="utf-8")

        changed_lines, diff = changed_line_count(before_source, candidate)
        (iteration_dir / "repair.diff").write_text(diff, encoding="utf-8")

        input_tokens = response.input_tokens or 0
        output_tokens = response.output_tokens or 0
        used_tokens = response.total_tokens or input_tokens + output_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_tokens += used_tokens
        total_latency += response.latency_seconds

        record = {
            "iteration": iteration,
            "failure_class_before": category,
            "evidence": evidence,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "tokens_used": response.total_tokens,
            "latency_seconds": response.latency_seconds,
            "changed_line_count": changed_lines,
            "host_validation_passed": validation_code == 0,
            "failure_class_after": (
                "none" if validation_code == 0 else classify_failure(validation_output)
            ),
        }
        iterations.append(record)
        (iteration_dir / "iteration.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )

        print(
            f"Iteration {iteration}/{max_iterations}: "
            f"tokens={response.total_tokens}, latency={response.latency_seconds:.2f}s, "
            f"host_passed={validation_code == 0}"
        )
        if validation_code == 0:
            success_iteration = iteration
            break

    protected_unchanged = all(
        protected_before[name] == sha256(workspace / name) for name in protected
    )

    independent = config["independent_validation"]
    independent_code: int | None = None
    independent_output = ""
    if validation_code == 0 and independent.get("enabled", False):
        command = [
            str(x).replace("{workspace}", str(workspace))
            for x in independent["command"]
        ]
        process = run(command, root)
        independent_code = process.returncode
        independent_output = process.stdout or ""
        (run_dir / "independent_validation.log").write_text(
            independent_output, encoding="utf-8"
        )

    final_source = (workspace / editable[0]).read_text(encoding="utf-8")
    final_changed_lines, final_diff = changed_line_count(initial_source, final_source)
    (run_dir / "repair.diff").write_text(final_diff, encoding="utf-8")

    passed = (
        success_iteration is not None
        and protected_unchanged
        and validation_code == 0
        and independent_code == 0
    )
    result = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp,
        "repair_mode": "iterative_direct_api",
        "provider": "siliconflow",
        "model": config["model"],
        "max_iterations": max_iterations,
        "iterations_executed": len(iterations),
        "success_iteration": success_iteration,
        "initial_failure_class": initial_failure_class,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_api_latency_seconds": round(total_latency, 3),
        "final_changed_line_count": final_changed_lines,
        "protected_files_unchanged": protected_unchanged,
        "post_host_validation_passed": validation_code == 0,
        "independent_validation_passed": (
            independent_code == 0 if independent_code is not None else False
        ),
        "passed": passed,
        "iterations": iterations,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    if not args.keep_workspace:
        shutil.rmtree(workspace)

    print(f"Experiment: {experiment_id}")
    print(f"Iterations executed: {len(iterations)}/{max_iterations}")
    print(f"Success iteration: {success_iteration}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total API latency: {total_latency:.2f}s")
    print(f"Post-repair host test passed: {validation_code == 0}")
    print(f"Independent validation passed: {independent_code == 0}")
    print(f"Results: {run_dir.relative_to(root)}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
