"""Stable, analysis-friendly artefacts for repair attempts and runs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from agent.budget import BudgetState
from agent.tools.reports import write_json


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
    "result": "result.json",
}


def _failure(result: dict[str, Any]) -> dict[str, Any]:
    feedback = result.get("feedback")
    return feedback if isinstance(feedback, dict) else {}


def _stage(
    status: str,
    *,
    reason: str | None = None,
    failure_class: str | None = None,
    evidence: list[str] | None = None,
    log_path: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run": status != "not_run",
        "status": status,
        "passed": True if status == "passed" else False if status == "failed" else None,
    }
    if reason:
        record["reason"] = reason
    if failure_class:
        record["failure_class"] = failure_class
    if evidence is not None:
        record["evidence"] = evidence
    if log_path:
        record["log_path"] = log_path
    return record


def _log_stage(
    attempt_dir: Path,
    filename: str,
    passed: bool | None,
    *,
    not_run_reason: str,
    failure_class: str | None = None,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    path = attempt_dir / filename
    if not path.is_file():
        return _stage("not_run", reason=not_run_reason)
    return _stage(
        "passed" if passed is True else "failed",
        failure_class=failure_class,
        evidence=evidence,
        log_path=filename,
    )


def _ensure_text(path: Path, content: str = "") -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _copy_or_empty(source: Path, destination: Path) -> None:
    if source.is_file():
        shutil.copy2(source, destination)
    else:
        destination.write_text("", encoding="utf-8")


def record_attempt_artifacts(
    *,
    attempt_dir: Path,
    editable_file: str,
    config: dict[str, Any],
    result: dict[str, Any],
    budget_enforced: bool,
) -> dict[str, Any]:
    """Materialise the same canonical record for every repair outcome."""
    attempt_dir.mkdir(parents=True, exist_ok=True)
    failure = _failure(result)
    failure_class = str(failure.get("failure_class", "none"))
    evidence = [str(item) for item in failure.get("evidence", [])]
    provider_failed = failure_class == "model_generation_error"
    output_rejected = failure_class == "invalid_model_output"
    candidate_accepted = not provider_failed and not output_rejected
    provenance = "accepted_model_output" if candidate_accepted else "last_valid_source"
    candidate_record_kind = (
        "generated_candidate" if candidate_accepted else "last_valid_source"
    )

    for filename in ("system_prompt.txt", "prompt.txt", "raw_response.txt"):
        _ensure_text(attempt_dir / filename)

    workspace_candidate = attempt_dir / "workspace" / editable_file
    if not workspace_candidate.is_file():
        raise FileNotFoundError(
            f"Attempt candidate source is unavailable: {workspace_candidate}"
        )
    shutil.copy2(workspace_candidate, attempt_dir / ATTEMPT_ARTIFACTS["candidate"])
    _copy_or_empty(
        attempt_dir / "repair.diff",
        attempt_dir / ATTEMPT_ARTIFACTS["diff"],
    )

    initial_diagnosis = result.get("diagnosis")
    final_diagnosis = result.get("final_diagnosis")
    diagnosis = {
        "schema_version": 1,
        "attempt": result.get("attempt"),
        "initial": initial_diagnosis if isinstance(initial_diagnosis, dict) else None,
        "final": final_diagnosis if isinstance(final_diagnosis, dict) else None,
        "active": (
            final_diagnosis
            if isinstance(final_diagnosis, dict)
            else initial_diagnosis
            if isinstance(initial_diagnosis, dict)
            else None
        ),
    }
    write_json(attempt_dir / ATTEMPT_ARTIFACTS["diagnosis"], diagnosis)

    output_validation = result.get("output_validation")
    if output_rejected:
        output_evidence = (
            [str(item) for item in output_validation.get("evidence", [])]
            if isinstance(output_validation, dict)
            else evidence
        )
        model_output_stage = _stage(
            "failed",
            failure_class="invalid_model_output",
            evidence=output_evidence,
        )
    elif provider_failed:
        model_output_stage = _stage(
            "not_run",
            reason="model_generation_failed_before_output_validation",
        )
    else:
        model_output_stage = _stage("passed", failure_class="none", evidence=[])

    if provider_failed:
        model_generation_stage = _stage(
            "failed",
            failure_class="model_generation_error",
            evidence=evidence,
        )
    else:
        model_generation_stage = _stage("passed", failure_class="none", evidence=[])

    if candidate_accepted:
        scope_stage = _stage(
            "passed" if result.get("editable_scope_respected") is True else "failed",
            failure_class=None if result.get("editable_scope_respected") is True else "scope_violation",
        )
        protected_stage = _stage(
            "passed" if result.get("protected_files_unchanged") is True else "failed",
            failure_class=(
                None
                if result.get("protected_files_unchanged") is True
                else "protected_file_modified"
            ),
        )
    else:
        scope_stage = _stage("not_run", reason="no_model_candidate_was_accepted")
        protected_stage = _stage("not_run", reason="no_model_candidate_was_accepted")

    post_reason = (
        "model_generation_failed"
        if provider_failed
        else "model_output_rejected"
        if output_rejected
        else "post_host_validation_not_reached"
    )
    independent_reason = (
        "independent_validation_disabled"
        if not config.get("independent_validation", {}).get("enabled", False)
        else "candidate_did_not_reach_independent_validation"
    )
    validation = {
        "schema_version": 1,
        "attempt": result.get("attempt"),
        "candidate_accepted": candidate_accepted,
        "candidate_provenance": provenance,
        "host_before": _log_stage(
            attempt_dir,
            "host_validation_before.log",
            result.get("pre_host_validation_passed"),
            not_run_reason="initial_host_validation_not_recorded",
            failure_class=str(result.get("failure_class", "unknown")),
            evidence=(
                [str(item) for item in initial_diagnosis.get("evidence", [])]
                if isinstance(initial_diagnosis, dict)
                else []
            ),
        ),
        "model_generation": model_generation_stage,
        "model_output_validation": model_output_stage,
        "editable_scope": scope_stage,
        "protected_files": protected_stage,
        "host_after": _log_stage(
            attempt_dir,
            "host_validation_after.log",
            result.get("post_host_validation_passed"),
            not_run_reason=post_reason,
            failure_class=(
                failure_class if failure.get("stage") == "host_validation" else None
            ),
            evidence=evidence if failure.get("stage") == "host_validation" else [],
        ),
        "independent_csim": _log_stage(
            attempt_dir,
            "independent_validation.log",
            result.get("independent_validation_passed"),
            not_run_reason=independent_reason,
            failure_class=(failure_class if failure.get("stage") == "csim" else None),
            evidence=evidence if failure.get("stage") == "csim" else [],
        ),
        "synthesis": _stage(
            "not_run",
            reason="synthesis_is_performed_after_repair_promotion",
        ),
        "cosim": _stage(
            "not_run",
            reason="cosim_is_performed_after_repair_promotion",
        ),
        "overall": {
            "passed": result.get("passed") is True,
            "failed_stage": None if result.get("passed") is True else failure.get("stage"),
            "failure_class": "none" if result.get("passed") is True else failure_class,
            "evidence": [] if result.get("passed") is True else evidence,
        },
    }
    write_json(attempt_dir / ATTEMPT_ARTIFACTS["validation"], validation)

    input_tokens = int(result.get("input_tokens") or 0)
    output_tokens = int(result.get("output_tokens") or 0)
    token_usage = {
        "schema_version": 1,
        "attempt": result.get("attempt"),
        "provider": result.get("provider", "siliconflow"),
        "model": result.get("model"),
        "temperature": config.get("temperature", 0.0),
        "thinking_budget": result.get("thinking_budget"),
        "thinking_enabled": False,
        "model_call_status": "failed" if provider_failed else "completed",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency_seconds": float(result.get("latency_seconds") or 0.0),
        "charged_to_task_budget": budget_enforced,
    }
    write_json(attempt_dir / ATTEMPT_ARTIFACTS["token_usage"], token_usage)

    result["artifact_schema_version"] = 1
    result["candidate_accepted"] = candidate_accepted
    result["candidate_provenance"] = provenance
    result["candidate_record_kind"] = candidate_record_kind
    result["validation"] = validation
    result["artifacts"] = dict(ATTEMPT_ARTIFACTS)
    write_json(attempt_dir / ATTEMPT_ARTIFACTS["result"], result)
    return result


def _usage_by_stage(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    usage: dict[str, dict[str, int]] = {}
    for event in events:
        stage = str(event.get("stage", "unknown"))
        bucket = usage.setdefault(stage, {})
        resource = str(event.get("resource", "unknown"))
        if resource == "tokens":
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                bucket[key] = bucket.get(key, 0) + int(event.get(key) or 0)
        else:
            bucket[resource] = bucket.get(resource, 0) + int(event.get("amount") or 0)
    return usage


def write_budget_summary(
    *,
    run_root: Path,
    budget: BudgetState | None,
    result: dict[str, Any],
) -> Path:
    """Write one run-level budget record, including unbounded test runs."""
    if budget is None:
        data: dict[str, Any] = {
            "schema_version": 1,
            "budget_enforced": False,
            "initial": None,
            "consumed": {
                "input_tokens": int(result.get("input_tokens") or 0),
                "output_tokens": int(result.get("output_tokens") or 0),
                "total_tokens": int(result.get("tokens_used") or 0),
            },
            "remaining": None,
            "stop_reason": None,
            "events": [],
            "usage_by_stage": {},
            "usage_by_phase": {"repair": {}},
        }
    else:
        data = budget.summary()
        data["budget_enforced"] = True
        data["usage_by_stage"] = _usage_by_stage(data.get("events", []))
        phase_usage: dict[str, int] = {}
        for values in data["usage_by_stage"].values():
            for key, value in values.items():
                phase_usage[key] = phase_usage.get(key, 0) + value
        data["usage_by_phase"] = {"repair": phase_usage}

    data.update(
        {
            "termination_reason": result.get("termination_reason"),
            "repair_passed": result.get("passed") is True,
            "attempt_count": int(result.get("attempt_count") or 0),
        }
    )
    path = run_root / "budget_summary.json"
    write_json(path, data)
    return path
