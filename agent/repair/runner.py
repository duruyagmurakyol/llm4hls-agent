"""Configuration-driven direct-API HLS repair workflow."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.budget import BudgetState
from agent.repair.diagnose import build_diagnosis, diagnose, format_diagnosis
from agent.repair.generate import generate_repair
from agent.state import ValidationResult
from agent.tools.command_runner import run_command
from agent.tools.reports import write_json
from agent.tools.validation import classify_failure, extract_evidence

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    config_path = Path(config).resolve()
    return json.loads(config_path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(root: Path, files: list[str]) -> dict[str, str]:
    return {name: _sha256(root / name) for name in files}


def _infer_top_function(config: dict[str, Any], workspace: Path) -> str | None:
    explicit = config.get("top_function")
    if explicit:
        return str(explicit)
    declaration = re.compile(
        r"\b(?:void|int|float|double|bool|long|short|unsigned|signed|auto)\s+"
        r"([A-Za-z_]\w*)\s*\("
    )
    for name in config.get("context_files", config.get("protected_files", [])):
        path = workspace / str(name)
        if path.suffix.lower() not in {".h", ".hh", ".hpp"} or not path.is_file():
            continue
        match = declaration.search(path.read_text(encoding="utf-8", errors="ignore"))
        if match:
            return match.group(1)
    return None


def _diagnosis(
    config: dict[str, Any],
    workspace: Path,
    validation: ValidationResult,
    *,
    stage: str,
) -> dict[str, object]:
    return diagnose(
        validation,
        stage=stage,
        editable_files=[str(item) for item in config.get("editable_files", [])],
        protected_files=[str(item) for item in config.get("protected_files", [])],
        top_function=_infer_top_function(config, workspace),
        repair_constraints=[str(item) for item in config.get("repair_constraints", [])],
    )


def _validate(config: dict[str, Any], workspace: Path) -> tuple[ValidationResult, str]:
    host = config["host_validation"]
    compiled = run_command([str(x) for x in host["command"]], cwd=workspace, echo=False)
    output = compiled.output
    result = compiled
    if compiled.passed:
        result = run_command([str(x) for x in host["run_command"]], cwd=workspace, echo=False)
        output += result.output
    return ValidationResult(
        passed=result.passed,
        failure_class=(
            "none"
            if result.passed
            else classify_failure(
                output,
                stage="host",
                timed_out=result.timed_out,
            )
        ),
        return_code=result.return_code,
        evidence=[] if result.passed else extract_evidence(output),
    ), output


def _feedback_diagnosis(
    config: dict[str, Any],
    workspace: Path,
    feedback: dict[str, Any],
) -> dict[str, object]:
    existing = feedback.get("diagnosis")
    if isinstance(existing, dict):
        return existing
    return build_diagnosis(
        stage=str(feedback.get("stage", "validation")),
        failure_class=str(feedback.get("failure_class", "unknown")),
        evidence=[str(item) for item in feedback.get("evidence", [])],
        editable_files=[str(item) for item in config.get("editable_files", [])],
        protected_files=[str(item) for item in config.get("protected_files", [])],
        top_function=_infer_top_function(config, workspace),
        repair_constraints=[str(item) for item in config.get("repair_constraints", [])],
    )


def _prompts(
    config: dict[str, Any],
    workspace: Path,
    validation: ValidationResult,
    feedback: dict[str, Any] | None = None,
) -> tuple[str, str]:
    editable = str(config["editable_files"][0])
    source = (workspace / editable).read_text(encoding="utf-8")
    contexts = []
    for name in config.get("context_files", config["protected_files"]):
        contexts.append(f"FILE: {name}\n```\n{(workspace / name).read_text(encoding='utf-8')}\n```")
    current_diagnosis = _diagnosis(
        config,
        workspace,
        validation,
        stage="host_validation",
    )
    system = (
        "You repair AMD/Xilinx HLS C++ code. Return only the complete repaired contents "
        "of the editable source file. Do not use Markdown fences, explanations, JSON, or patches. "
        "Preserve the declared top-function interface and make the smallest necessary repair."
    )
    retry = ""
    if feedback:
        retry_diagnosis = _feedback_diagnosis(config, workspace, feedback)
        retry = (
            f"Previous repair attempt {feedback.get('attempt')} failed.\n"
            "Previous structured diagnosis:\n"
            f"{format_diagnosis(retry_diagnosis)}\n"
            "The editable file below is the previous candidate. Correct it without repeating the failed change.\n\n"
        )
    user = (
        retry
        + "Current structured diagnosis:\n"
        + format_diagnosis(current_diagnosis)
        + "\n\n"
        + f"EDITABLE FILE: {editable}\n```\n{source}\n```\n\n"
        + "\n\n".join(contexts)
        + "\n\nReturn only the full repaired editable file."
    )
    return system, user


def _feedback(
    attempt: int,
    diagnosis: dict[str, object],
    *,
    return_code: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempt": attempt,
        "stage": diagnosis["stage"],
        "failure_class": diagnosis["failure_class"],
        "evidence": diagnosis["evidence"],
        "diagnosis": diagnosis,
    }
    if return_code is not None:
        result["return_code"] = return_code
    return result


def _failure_feedback(
    attempt: int,
    *,
    config: dict[str, Any],
    workspace: Path,
    scope_ok: bool,
    protected_unchanged: bool,
    post_validation: ValidationResult,
    independent_code: int | None,
    independent_output: str,
    independent_timed_out: bool,
) -> dict[str, Any]:
    editable = [str(item) for item in config.get("editable_files", [])]
    protected = [str(item) for item in config.get("protected_files", [])]
    common = {
        "editable_files": editable,
        "protected_files": protected,
        "top_function": _infer_top_function(config, workspace),
        "repair_constraints": [str(item) for item in config.get("repair_constraints", [])],
    }
    if not scope_ok:
        diagnosis = build_diagnosis(
            stage="editable_scope",
            failure_class="scope_violation",
            evidence=["The response modified a file outside repair.editable_files."],
            **common,
        )
        return _feedback(attempt, diagnosis)
    if not protected_unchanged:
        diagnosis = build_diagnosis(
            stage="protected_files",
            failure_class="protected_file_modified",
            evidence=["A protected source, header, testbench, or build file changed."],
            **common,
        )
        return _feedback(attempt, diagnosis)
    if not post_validation.passed:
        diagnosis = _diagnosis(
            config,
            workspace,
            post_validation,
            stage="host_validation",
        )
        return _feedback(attempt, diagnosis, return_code=post_validation.return_code)

    evidence = extract_evidence(independent_output)
    failure_class = classify_failure(
        independent_output,
        stage="csim",
        timed_out=independent_timed_out,
    )
    diagnosis = build_diagnosis(
        stage="csim",
        failure_class=failure_class,
        evidence=evidence,
        **common,
    )
    return _feedback(attempt, diagnosis, return_code=independent_code)


def _run_repair_once(
    config_source: dict[str, Any] | str | Path,
    *,
    run_dir: Path | None = None,
    attempt: int = 1,
    seed_source: Path | None = None,
    feedback: dict[str, Any] | None = None,
    keep_workspace: bool = False,
    budget: BudgetState | None = None,
) -> tuple[bool, Path, dict[str, Any]]:
    config = _load_config(config_source)
    if config.get("repair_mode") != "direct_api":
        raise ValueError("Config must use repair_mode=direct_api")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment_id = str(config["experiment_id"])
    run_dir = run_dir or REPO_ROOT / "results" / "experiments" / experiment_id / timestamp
    workspace = run_dir / "workspace"
    benchmark_source = REPO_ROOT / config["benchmark_source"]
    if not benchmark_source.is_dir():
        raise FileNotFoundError(f"Benchmark source not found: {benchmark_source}")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(benchmark_source, workspace)
    fault_metadata = workspace / "fault.txt"
    if fault_metadata.exists():
        fault_metadata.unlink()

    editable = [str(x) for x in config["editable_files"]]
    protected = [str(x) for x in config["protected_files"]]
    tracked = editable + protected
    if seed_source is not None:
        shutil.copy2(seed_source, workspace / editable[0])

    before_hashes = _hashes(workspace, tracked)
    before_source = (workspace / editable[0]).read_text(encoding="utf-8")
    (run_dir / "before.cpp").write_text(before_source, encoding="utf-8")

    pre_validation, pre_output = _validate(config, workspace)
    pre_diagnosis = _diagnosis(
        config,
        workspace,
        pre_validation,
        stage="host_validation",
    )
    (run_dir / "host_validation_before.log").write_text(pre_output, encoding="utf-8")
    write_json(run_dir / "diagnosis_before.json", pre_diagnosis)
    system_prompt, user_prompt = _prompts(config, workspace, pre_validation, feedback)
    (run_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    (run_dir / "prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")

    thinking_budget_value = config.get("thinking_budget")
    stage = f"repair_attempt_{attempt:03d}_generation"
    if budget is not None:
        budget.charge_iteration(stage=f"repair_attempt_{attempt:03d}")
        budget.charge_model_call(stage=stage)
    try:
        repaired, response = generate_repair(
            model=str(config["model"]),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_output_tokens", 2048)),
            timeout_seconds=int(config.get("api_timeout_seconds", 120)),
            thinking_budget=int(thinking_budget_value) if thinking_budget_value is not None else None,
        )
    except Exception:
        if budget is not None:
            budget.update_last_event(success=False)
        raise
    if budget is not None:
        budget.update_last_event(success=True)
        budget.record_model_tokens(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            stage=stage,
        )

    (run_dir / "raw_response.txt").write_text(response.content, encoding="utf-8")
    write_json(run_dir / "api_response.json", response.raw_response)
    (workspace / editable[0]).write_text(repaired, encoding="utf-8")

    after_hashes = _hashes(workspace, tracked)
    modified = [name for name in tracked if before_hashes[name] != after_hashes[name]]
    protected_unchanged = all(before_hashes[name] == after_hashes[name] for name in protected)
    scope_ok = set(modified).issubset(set(editable))
    diff = "".join(
        difflib.unified_diff(
            before_source.splitlines(keepends=True),
            repaired.splitlines(keepends=True),
            fromfile=f"before/{editable[0]}",
            tofile=f"after/{editable[0]}",
        )
    )
    (run_dir / "repair.diff").write_text(diff, encoding="utf-8")
    changed_lines = sum(
        1 for line in diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )

    post_validation, post_output = _validate(config, workspace)
    (run_dir / "host_validation_after.log").write_text(post_output, encoding="utf-8")

    independent = config["independent_validation"]
    independent_code: int | None = None
    independent_output = ""
    independent_timed_out = False
    if independent.get("enabled", False):
        validation_stage = f"repair_attempt_{attempt:03d}_independent_validation"
        if budget is not None:
            budget.charge_csim(stage=validation_stage)
        command = [str(x).replace("{workspace}", str(workspace)) for x in independent["command"]]
        process = run_command(command, cwd=REPO_ROOT, echo=False)
        independent_code = process.return_code
        independent_output = process.output
        independent_timed_out = process.timed_out
        if budget is not None:
            budget.update_last_event(success=process.passed)
        (run_dir / "independent_validation.log").write_text(independent_output, encoding="utf-8")

    independent_passed = independent_code == 0 if independent.get("enabled", False) else True
    passed = scope_ok and protected_unchanged and post_validation.passed and independent_passed
    failure = None if passed else _failure_feedback(
        attempt,
        config=config,
        workspace=workspace,
        scope_ok=scope_ok,
        protected_unchanged=protected_unchanged,
        post_validation=post_validation,
        independent_code=independent_code,
        independent_output=independent_output,
        independent_timed_out=independent_timed_out,
    )
    final_diagnosis = failure.get("diagnosis") if isinstance(failure, dict) else None
    if isinstance(final_diagnosis, dict):
        write_json(run_dir / "diagnosis_after.json", final_diagnosis)

    result = {
        "schema_version": 4,
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp,
        "attempt": attempt,
        "repair_mode": "direct_api",
        "provider": "siliconflow",
        "model": config["model"],
        "thinking_budget": thinking_budget_value,
        "failure_class": pre_validation.failure_class,
        "diagnosis": pre_diagnosis,
        "final_diagnosis": final_diagnosis,
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
        "independent_validation_passed": independent_passed,
        "repair_diff_present": bool(diff.strip()),
        "passed": passed,
        "feedback": failure,
    }
    write_json(run_dir / "result.json", result)
    if not keep_workspace:
        shutil.rmtree(workspace)
    return passed, run_dir, result


def run_repair(
    config_source: dict[str, Any] | str | Path,
    *,
    keep_workspace: bool = False,
    budget: BudgetState | None = None,
) -> tuple[bool, Path, dict[str, Any]]:
    from agent.repair.retry import run_repair_loop

    return run_repair_loop(
        config_source,
        keep_workspace=keep_workspace,
        budget=budget,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a direct-API HLS repair task.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()
    passed, run_dir, result = run_repair(args.config, keep_workspace=args.keep_workspace)
    print(f"Experiment: {result['experiment_id']}")
    print(f"Model: {result['model']}")
    print(f"Attempts: {result['attempt_count']}")
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
