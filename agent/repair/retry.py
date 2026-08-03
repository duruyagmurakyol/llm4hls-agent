"""Budget-bounded retry loop for direct HLS repair attempts."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.budget import BudgetExceeded, BudgetState
from agent.repair.diagnose import build_diagnosis
from agent.repair.output_validation import InvalidModelOutputError
from agent.repair.strategy import (
    DO_NOT_REPEAT_CONSTRAINT,
    build_strategy,
    strategy_feedback_evidence,
)
from agent.tools.reports import load_json, write_json
from agent.tools.validation import classify_failure


ATTEMPT_ARTIFACTS = {
    "diagnosis": "diagnosis.json",
    "system_prompt": "system_prompt.txt",
    "prompt": "prompt.txt",
    "raw_response": "raw_response.txt",
    "candidate": "candidate.cpp",
    "diff": "diff.patch",
    "strategy": "strategy.json",
    "validation": "validation.json",
    "token_usage": "token_usage.json",
}


def _can_attempt(config: dict[str, Any], budget: BudgetState | None) -> bool:
    if budget is None:
        return True
    return budget.can_generate_candidate(
        reserve_csim=1 if config["independent_validation"].get("enabled", False) else 0,
        reserve_synthesis=0,
        reserve_cosim=0,
    )


def _attempt_limit(config: dict[str, Any], budget: BudgetState | None) -> int:
    configured = config.get("max_attempts")
    if configured is not None:
        value = int(configured)
        if value <= 0:
            raise ValueError("max_attempts must be a positive integer")
        return value
    if budget is None:
        return 1
    limits = [budget.remaining("iterations"), budget.remaining("model_calls")]
    if config["independent_validation"].get("enabled", False):
        limits.append(budget.remaining("csim_calls"))
    return min(limits)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _record_strategy(
    *,
    attempt_dir: Path,
    editable: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Persist the attempted source edit and enrich retry feedback with it."""
    before_path = attempt_dir / "before.cpp"
    candidate_path = attempt_dir / "workspace" / editable
    before_source = before_path.read_text(encoding="utf-8") if before_path.is_file() else ""
    candidate_source = (
        candidate_path.read_text(encoding="utf-8") if candidate_path.is_file() else before_source
    )
    strategy = build_strategy(
        before_source=before_source,
        candidate_source=candidate_source,
        editable_file=editable,
    )
    write_json(attempt_dir / "strategy.json", strategy)
    result["strategy"] = strategy

    feedback = result.get("feedback")
    if isinstance(feedback, dict):
        feedback["strategy"] = strategy
        strategy_evidence = strategy_feedback_evidence(strategy)
        feedback["evidence"] = _unique(
            [str(item) for item in feedback.get("evidence", [])] + strategy_evidence
        )

        diagnosis = feedback.get("diagnosis")
        if isinstance(diagnosis, dict):
            diagnosis["evidence"] = _unique(
                [str(item) for item in diagnosis.get("evidence", [])] + strategy_evidence
            )
            diagnosis["repair_constraints"] = _unique(
                [str(item) for item in diagnosis.get("repair_constraints", [])]
                + [DO_NOT_REPEAT_CONSTRAINT]
            )

    write_json(attempt_dir / "result.json", result)
    return strategy


def _stage_record(
    *,
    ran: bool,
    passed: bool | None,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run": ran,
        "passed": passed if ran else None,
    }
    if reason:
        record["reason"] = reason
    if details:
        record["details"] = details
    return record


def _write_attempt_contract(
    *,
    attempt_dir: Path,
    editable: str,
    result: dict[str, Any],
) -> None:
    """Materialise the small, stable FPT-507 artefact contract."""
    workspace_candidate = attempt_dir / "workspace" / editable
    if not workspace_candidate.is_file():
        raise FileNotFoundError(f"Attempt candidate not found: {workspace_candidate}")

    raw_response = attempt_dir / ATTEMPT_ARTIFACTS["raw_response"]
    raw_response.touch(exist_ok=True)
    shutil.copy2(workspace_candidate, attempt_dir / ATTEMPT_ARTIFACTS["candidate"])

    legacy_diff = attempt_dir / "repair.diff"
    canonical_diff = attempt_dir / ATTEMPT_ARTIFACTS["diff"]
    if legacy_diff.is_file():
        shutil.copy2(legacy_diff, canonical_diff)
    else:
        canonical_diff.write_text("", encoding="utf-8")

    write_json(
        attempt_dir / ATTEMPT_ARTIFACTS["diagnosis"],
        {
            "initial": result.get("diagnosis"),
            "final": result.get("final_diagnosis"),
        },
    )

    output_validation = result.get("output_validation")
    invalid_output = isinstance(output_validation, dict)
    generation_error = result.get("generation_error")
    provider_error = generation_error is not None and not invalid_output
    host_before_ran = (attempt_dir / "host_validation_before.log").is_file()
    host_after_ran = (attempt_dir / "host_validation_after.log").is_file()
    independent_ran = (attempt_dir / "independent_validation.log").is_file()

    validation = {
        "model_generation": _stage_record(
            ran=True,
            passed=not provider_error,
            reason=str(generation_error) if provider_error else None,
        ),
        "model_output_validation": _stage_record(
            ran=not provider_error,
            passed=False if invalid_output else True if not provider_error else None,
            reason="provider_error" if provider_error else None,
            details=output_validation if invalid_output else None,
        ),
        "host_before": _stage_record(
            ran=host_before_ran,
            passed=result.get("pre_host_validation_passed"),
        ),
        "host_after": _stage_record(
            ran=host_after_ran,
            passed=result.get("post_host_validation_passed"),
            reason=None if host_after_ran else "generation_did_not_produce_an_accepted_candidate",
        ),
        "independent_csim": _stage_record(
            ran=independent_ran,
            passed=result.get("independent_validation_passed"),
            reason=None if independent_ran else "candidate_did_not_reach_independent_validation",
        ),
    }
    write_json(attempt_dir / ATTEMPT_ARTIFACTS["validation"], validation)

    write_json(
        attempt_dir / ATTEMPT_ARTIFACTS["token_usage"],
        {
            "provider": result.get("provider"),
            "model": result.get("model"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "total_tokens": result.get("tokens_used"),
            "latency_seconds": result.get("latency_seconds"),
            "thinking_budget": result.get("thinking_budget"),
        },
    )

    result["candidate_record_kind"] = (
        "last_valid_source" if provider_error or invalid_output else "generated_candidate"
    )
    result["validation"] = validation
    result["artifacts"] = dict(ATTEMPT_ARTIFACTS)
    write_json(attempt_dir / "result.json", result)


def _write_budget_summary(
    *,
    run_root: Path,
    budget: BudgetState | None,
    termination_reason: str,
) -> None:
    if budget is None:
        summary: dict[str, Any] = {
            "schema_version": 1,
            "available": False,
            "reason": "No BudgetState was supplied to this repair run.",
        }
    else:
        summary = budget.summary()
    summary["termination_reason"] = termination_reason
    write_json(run_root / "budget_summary.json", summary)


def _exception_attempt(
    config: dict[str, Any],
    *,
    repo_root: Path,
    attempt_dir: Path,
    attempt: int,
    seed_source: Path | None,
    error: Exception,
) -> tuple[bool, Path, dict[str, Any]]:
    """Turn a provider or output-validation exception into retryable evidence."""
    workspace = attempt_dir / "workspace"
    editable = str(config["editable_files"][0])
    if not workspace.is_dir():
        benchmark_source = repo_root / str(config["benchmark_source"])
        shutil.copytree(benchmark_source, workspace)
        fault_metadata = workspace / "fault.txt"
        if fault_metadata.exists():
            fault_metadata.unlink()
        if seed_source is not None:
            shutil.copy2(seed_source, workspace / editable)

    candidate = workspace / editable
    if not candidate.is_file():
        raise RuntimeError(
            "Repair attempt failed before an editable candidate workspace was available"
        ) from error

    invalid_output = isinstance(error, InvalidModelOutputError)
    error_text = f"{type(error).__name__}: {error}"
    stage = "model_output_validation" if invalid_output else "model_generation"
    failure_class = "invalid_model_output" if invalid_output else "model_generation_error"
    evidence = (
        [str(item) for item in error.report.get("evidence", [])]
        if invalid_output
        else [error_text]
    )

    if invalid_output:
        (attempt_dir / "raw_response.txt").write_text(
            error.raw_response,
            encoding="utf-8",
        )
        write_json(attempt_dir / "api_response.json", error.raw_api_response)
        write_json(attempt_dir / "output_validation.json", error.report)
        (attempt_dir / "model_output_validation_error.log").write_text(
            error_text + "\n",
            encoding="utf-8",
        )
    else:
        (attempt_dir / "model_generation_error.log").write_text(
            error_text + "\n",
            encoding="utf-8",
        )

    pre_log = attempt_dir / "host_validation_before.log"
    pre_output = pre_log.read_text(encoding="utf-8") if pre_log.is_file() else ""
    diagnosis = build_diagnosis(
        stage=stage,
        failure_class=failure_class,
        evidence=evidence,
        editable_files=[str(item) for item in config.get("editable_files", [])],
        protected_files=[str(item) for item in config.get("protected_files", [])],
        top_function=str(config["top_function"]) if config.get("top_function") else None,
        repair_constraints=[str(item) for item in config.get("repair_constraints", [])],
    )
    write_json(attempt_dir / "diagnosis_after.json", diagnosis)
    feedback = {
        "attempt": attempt,
        "stage": stage,
        "failure_class": failure_class,
        "evidence": evidence,
        "diagnosis": diagnosis,
    }

    initial_diagnosis_path = attempt_dir / "diagnosis_before.json"
    initial_diagnosis = (
        load_json(initial_diagnosis_path) if initial_diagnosis_path.is_file() else None
    )
    input_tokens = error.input_tokens if invalid_output else 0
    output_tokens = error.output_tokens if invalid_output else 0
    total_tokens = error.total_tokens if invalid_output else 0
    latency_seconds = error.latency_seconds if invalid_output else 0.0

    result = {
        "schema_version": 4,
        "experiment_id": str(config["experiment_id"]),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "attempt": attempt,
        "repair_mode": "direct_api",
        "provider": "siliconflow",
        "model": config["model"],
        "thinking_budget": config.get("thinking_budget"),
        "failure_class": classify_failure(pre_output) if pre_output else "unknown",
        "diagnosis": initial_diagnosis,
        "final_diagnosis": diagnosis,
        "pre_host_validation_passed": False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_used": total_tokens,
        "latency_seconds": latency_seconds,
        "modified_files": [],
        "protected_files_unchanged": True,
        "editable_scope_respected": True,
        "changed_line_count": 0,
        "tokens_per_changed_line": None,
        "post_host_validation_passed": False,
        "independent_validation_passed": False,
        "repair_diff_present": False,
        "passed": False,
        "generation_error": error_text,
        "output_validation": error.report if invalid_output else None,
        "feedback": feedback,
    }
    write_json(attempt_dir / "result.json", result)
    return False, attempt_dir, result


def run_repair_loop(
    config_source: dict[str, Any] | str | Path,
    *,
    keep_workspace: bool = False,
    budget: BudgetState | None = None,
) -> tuple[bool, Path, dict[str, Any]]:
    from agent.repair.runner import REPO_ROOT, _load_config, _run_repair_once, _sha256

    config = _load_config(config_source)
    maximum = _attempt_limit(config, budget)
    if maximum <= 0:
        raise RuntimeError("Repair could not start because no attempt budget was available")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = REPO_ROOT / "results" / "experiments" / str(config["experiment_id"]) / timestamp
    run_root.mkdir(parents=True)

    attempts: list[dict[str, Any]] = []
    attempt_results: list[dict[str, Any]] = []
    seed_source: Path | None = None
    feedback: dict[str, Any] | None = None
    final_dir = run_root
    passed = False
    editable = str(config["editable_files"][0])

    for attempt in range(1, maximum + 1):
        if not _can_attempt(config, budget):
            if budget is not None:
                budget.set_stop_reason("repair_budget_exhausted")
            break
        attempt_dir = run_root / f"attempt_{attempt:03d}"
        try:
            passed, final_dir, result = _run_repair_once(
                config,
                run_dir=attempt_dir,
                attempt=attempt,
                seed_source=seed_source,
                feedback=feedback,
                keep_workspace=True,
                budget=budget,
            )
        except BudgetExceeded:
            raise
        except Exception as error:
            if isinstance(error, InvalidModelOutputError) and budget is not None:
                budget.record_model_tokens(
                    input_tokens=error.input_tokens,
                    output_tokens=error.output_tokens,
                    stage=f"repair_attempt_{attempt:03d}_generation",
                )
            passed, final_dir, result = _exception_attempt(
                config,
                repo_root=REPO_ROOT,
                attempt_dir=attempt_dir,
                attempt=attempt,
                seed_source=seed_source,
                error=error,
            )

        strategy = _record_strategy(
            attempt_dir=final_dir,
            editable=editable,
            result=result,
        )
        _write_attempt_contract(
            attempt_dir=final_dir,
            editable=editable,
            result=result,
        )
        attempt_results.append(result)
        feedback = result.get("feedback")
        if not passed and not isinstance(feedback, dict):
            raise RuntimeError("Failed repair attempt did not produce structured feedback")
        candidate = final_dir / "workspace" / editable
        attempts.append(
            {
                "attempt": attempt,
                "run_dir": str(final_dir.relative_to(REPO_ROOT)),
                "passed": passed,
                "failed_stage": None if passed else feedback.get("stage"),
                "failure_class": "none" if passed else feedback.get("failure_class"),
                "evidence": [] if passed else feedback.get("evidence", []),
                "diagnosis": None if passed else feedback.get("diagnosis"),
                "strategy": strategy,
                "candidate_hash": _sha256(candidate),
                "artifacts": dict(ATTEMPT_ARTIFACTS),
            }
        )
        if passed:
            break
        seed_source = candidate

    if not attempt_results:
        raise RuntimeError("Repair could not start because no attempt budget was available")
    if not passed and budget is not None and not _can_attempt(config, budget):
        budget.set_stop_reason("repair_budget_exhausted")

    final = attempt_results[-1]
    total_input = sum(int(item.get("input_tokens") or 0) for item in attempt_results)
    total_output = sum(int(item.get("output_tokens") or 0) for item in attempt_results)
    termination_reason = (
        "repair_validated"
        if passed
        else "repair_budget_exhausted"
        if budget is not None and not _can_attempt(config, budget)
        else "repair_attempt_limit_reached"
    )
    result = {
        **final,
        "schema_version": 4,
        "failure_class": attempt_results[0]["failure_class"],
        "initial_diagnosis": attempt_results[0].get("diagnosis"),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "tokens_used": total_input + total_output,
        "termination_reason": termination_reason,
    }
    write_json(run_root / "repair_attempts.json", result)
    write_json(final_dir / "result.json", result)
    _write_budget_summary(
        run_root=run_root,
        budget=budget,
        termination_reason=termination_reason,
    )

    if not keep_workspace:
        for item in attempts:
            shutil.rmtree(REPO_ROOT / item["run_dir"] / "workspace")
    return passed, final_dir, result
