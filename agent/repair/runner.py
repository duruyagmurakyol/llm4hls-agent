"""Configuration-driven direct-API HLS repair workflow."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.repair.diagnose import diagnose
from agent.repair.generate import generate_repair
from agent.state import ValidationResult
from agent.tools.command_runner import run_command
from agent.tools.reports import write_json
from agent.tools.validation import classify_failure, extract_evidence

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(root: Path, files: list[str]) -> dict[str, str]:
    return {name: _sha256(root / name) for name in files}


def _validate(config: dict[str, Any], workspace: Path) -> tuple[ValidationResult, str]:
    host = config["host_validation"]
    compiled = run_command([str(x) for x in host["command"]], cwd=workspace, echo=False)
    output = compiled.output
    return_code = compiled.return_code
    if compiled.passed:
        executed = run_command([str(x) for x in host["run_command"]], cwd=workspace, echo=False)
        output += executed.output
        return_code = executed.return_code
    return ValidationResult(
        passed=return_code == 0,
        failure_class="none" if return_code == 0 else classify_failure(output),
        return_code=return_code,
        evidence=[] if return_code == 0 else extract_evidence(output),
    ), output


def _prompts(config: dict[str, Any], workspace: Path, validation: ValidationResult) -> tuple[str, str]:
    editable = str(config["editable_files"][0])
    source = (workspace / editable).read_text(encoding="utf-8")
    contexts = []
    for name in config.get("context_files", config["protected_files"]):
        contexts.append(f"FILE: {name}\n```\n{(workspace / name).read_text(encoding='utf-8')}\n```")
    diagnosis = diagnose(validation)
    system = (
        "You repair AMD/Xilinx HLS C++ code. Return only the complete repaired contents "
        "of the editable source file. Do not use Markdown fences, explanations, JSON, or patches. "
        "Preserve the declared top-function interface and make the smallest necessary repair."
    )
    user = (
        f"Failure class: {diagnosis['failure_class']}\n"
        f"Failure evidence:\n" + "\n".join(validation.evidence) + "\n\n"
        f"EDITABLE FILE: {editable}\n```\n{source}\n```\n\n"
        + "\n\n".join(contexts)
        + "\n\nReturn only the full repaired editable file."
    )
    return system, user


def run_repair(config_path: Path, *, keep_workspace: bool = False) -> tuple[bool, Path, dict[str, Any]]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("repair_mode") != "direct_api":
        raise ValueError("Config must use repair_mode=direct_api")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = str(config["experiment_id"])
    run_dir = REPO_ROOT / "results" / "experiments" / experiment_id / timestamp
    workspace = run_dir / "workspace"
    benchmark_source = REPO_ROOT / config["benchmark_source"]
    if not benchmark_source.is_dir():
        raise FileNotFoundError(f"Benchmark source not found: {benchmark_source}")
    run_dir.mkdir(parents=True)
    shutil.copytree(benchmark_source, workspace)
    fault_metadata = workspace / "fault.txt"
    if fault_metadata.exists():
        fault_metadata.unlink()

    editable = [str(x) for x in config["editable_files"]]
    protected = [str(x) for x in config["protected_files"]]
    tracked = editable + protected
    before_hashes = _hashes(workspace, tracked)
    before_source = (workspace / editable[0]).read_text(encoding="utf-8")
    (run_dir / "before.cpp").write_text(before_source, encoding="utf-8")

    pre_validation, pre_output = _validate(config, workspace)
    (run_dir / "host_validation_before.log").write_text(pre_output, encoding="utf-8")
    system_prompt, user_prompt = _prompts(config, workspace, pre_validation)
    (run_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    (run_dir / "prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")

    thinking_budget_value = config.get("thinking_budget")
    repaired, response = generate_repair(
        model=str(config["model"]),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_output_tokens", 2048)),
        timeout_seconds=int(config.get("api_timeout_seconds", 120)),
        thinking_budget=int(thinking_budget_value) if thinking_budget_value is not None else None,
    )
    (run_dir / "raw_response.txt").write_text(response.content, encoding="utf-8")
    write_json(run_dir / "api_response.json", response.raw_response)
    (workspace / editable[0]).write_text(repaired, encoding="utf-8")

    after_hashes = _hashes(workspace, tracked)
    modified = [name for name in tracked if before_hashes[name] != after_hashes[name]]
    protected_unchanged = all(before_hashes[name] == after_hashes[name] for name in protected)
    scope_ok = set(modified).issubset(set(editable))
    diff = "".join(difflib.unified_diff(
        before_source.splitlines(keepends=True), repaired.splitlines(keepends=True),
        fromfile=f"before/{editable[0]}", tofile=f"after/{editable[0]}",
    ))
    (run_dir / "repair.diff").write_text(diff, encoding="utf-8")
    changed_lines = sum(1 for line in diff.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))

    post_validation, post_output = _validate(config, workspace)
    (run_dir / "host_validation_after.log").write_text(post_output, encoding="utf-8")

    independent = config["independent_validation"]
    independent_code: int | None = None
    if independent.get("enabled", False):
        command = [str(x).replace("{workspace}", str(workspace)) for x in independent["command"]]
        process = run_command(command, cwd=REPO_ROOT, echo=False)
        independent_code = process.return_code
        (run_dir / "independent_validation.log").write_text(process.output, encoding="utf-8")

    result = {
        "schema_version": 3,
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp,
        "repair_mode": "direct_api",
        "provider": "siliconflow",
        "model": config["model"],
        "thinking_budget": thinking_budget_value,
        "failure_class": pre_validation.failure_class,
        "pre_host_validation_passed": pre_validation.passed,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "tokens_used": response.total_tokens,
        "latency_seconds": response.latency_seconds,
        "modified_files": modified,
        "protected_files_unchanged": protected_unchanged,
        "editable_scope_respected": scope_ok,
        "changed_line_count": changed_lines,
        "tokens_per_changed_line": response.total_tokens / changed_lines if response.total_tokens is not None and changed_lines else None,
        "post_host_validation_passed": post_validation.passed,
        "independent_validation_passed": independent_code == 0 if independent_code is not None else None,
        "repair_diff_present": bool(diff.strip()),
    }
    write_json(run_dir / "result.json", result)

    if not keep_workspace:
        shutil.rmtree(workspace)

    passed = (
        not pre_validation.passed
        and scope_ok
        and protected_unchanged
        and post_validation.passed
        and result["independent_validation_passed"] is True
    )
    return passed, run_dir, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one direct-API HLS repair task.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()
    passed, run_dir, result = run_repair(args.config, keep_workspace=args.keep_workspace)
    print(f"Experiment: {result['experiment_id']}")
    print(f"Model: {result['model']}")
    print(f"Failure class: {result['failure_class']}")
    print(f"Tokens: {result['tokens_used']} (input={result['input_tokens']}, output={result['output_tokens']})")
    print(f"Latency: {result['latency_seconds']:.2f}s")
    print(f"Modified files: {', '.join(result['modified_files']) if result['modified_files'] else 'none'}")
    print(f"Post-repair host test passed: {result['post_host_validation_passed']}")
    print(f"Independent validation passed: {result['independent_validation_passed']}")
    print(f"Results: {run_dir.relative_to(REPO_ROOT)}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
