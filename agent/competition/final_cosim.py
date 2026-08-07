"""Ranked post-search C/RTL co-simulation promotion gate.

Search uses static validation, CSim and synthesis to construct and rank a Pareto
set. After search, this module co-simulates only the highest-ranked selected
design. If it fails, it tries the next ranked Pareto candidate and finally the
verified baseline, stopping as soon as one design passes. This preserves final
correctness while avoiding unnecessary co-simulation runs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from agent.config import TaskManifest
from agent.optimise.selection import deterministic_selection_key
from agent.state import AgentResult, TrajectoryEvent
from agent.tools.cosim import run_cosim

REPO_ROOT = Path(__file__).resolve().parent.parent
CosimRunner = Callable[[TaskManifest, Path], dict[str, Any]]
POLICY_NAME = "ranked_selected_then_fallback"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(record: dict[str, Any]) -> Path | None:
    for key in ("archived_file", "candidate_file", "source"):
        value = record.get(key)
        if isinstance(value, str) and value:
            path = _resolve(value)
            if path.is_file():
                return path
    return None


def _validation(record: dict[str, Any]) -> dict[str, Any]:
    nested = record.get("validation")
    nested = dict(nested) if isinstance(nested, dict) else {}
    return {
        "static_validation": nested.get(
            "static_validation", record.get("static_validation", True)
        ),
        "csim": nested.get("csim", record.get("csim")),
        "synthesis": nested.get("synthesis", record.get("synthesis")),
        "cosim": nested.get("cosim", record.get("cosim")),
    }


def _identity(record: dict[str, Any], source: Path) -> str:
    digest = record.get("candidate_hash")
    if isinstance(digest, str) and digest:
        return digest
    return _hash(source)


def _baseline_record(output_dir: Path) -> dict[str, Any] | None:
    baseline = _load_json(output_dir / "verified_baseline.json")
    source = baseline.get("source")
    if not isinstance(source, str) or not _resolve(source).is_file():
        return None
    validation = baseline.get("validation")
    validation = dict(validation) if isinstance(validation, dict) else {}
    return {
        "role": "verified_baseline_fallback",
        "candidate_index": 0,
        "candidate_file": source,
        "archived_file": source,
        "candidate_hash": baseline.get("candidate_hash"),
        "metrics": dict(baseline.get("metrics") or {}),
        "fully_verified": bool(
            validation.get("csim_passed") is True
            and validation.get("synthesis_passed") is True
            and validation.get("cosim_passed") is True
        ),
        "meets_frequency_requirement": None,
        "meets_resource_limits": True,
        "resource_limit_compliance": {"configured": False, "passed": True},
        "cost": {},
        "verdict": "verified_baseline",
        "validation": {
            "static_validation": True,
            "csim": validation.get("csim_passed"),
            "synthesis": validation.get("synthesis_passed"),
            "cosim": validation.get("cosim_passed"),
        },
    }


def _collect_records(
    state: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], set[str], str | None]:
    selected = state.get("selected_design")
    selected_identity: str | None = None
    if isinstance(selected, dict):
        selected_source = _source(selected)
        if selected_source is not None:
            selected_identity = _identity(selected, selected_source)

    records: list[dict[str, Any]] = []
    pareto_identities: set[str] = set()
    seen: set[str] = set()

    def add(value: Any, *, pareto: bool = False) -> None:
        if not isinstance(value, dict):
            return
        record = dict(value)
        source = _source(record)
        if source is None:
            return
        identity = _identity(record, source)
        record["candidate_hash"] = identity
        record.setdefault("candidate_file", _display(source))
        record.setdefault("archived_file", _display(source))
        if identity not in seen:
            records.append(record)
            seen.add(identity)
        if pareto:
            pareto_identities.add(identity)

    add(selected)
    for item in state.get("pareto_archive", []):
        add(item, pareto=True)
    add(state.get("best_ppa_candidate"), pareto=True)
    add(state.get("best_correct_candidate"))
    add(state.get("original_baseline"))
    add(_baseline_record(output_dir))
    return records, pareto_identities, selected_identity


def _update_record(record: dict[str, Any], report: dict[str, Any]) -> None:
    passed = report.get("passed") is True
    validation = _validation(record)
    validation["cosim"] = passed
    record["validation"] = validation
    record["cosim_required"] = True
    record["cosim"] = passed
    record["cosim_run"] = True
    record["fully_verified"] = bool(
        validation.get("static_validation") is True
        and validation.get("csim") is True
        and validation.get("synthesis") is True
        and passed
    )
    record["final_cosim"] = {
        "passed": passed,
        "failure_class": report.get("failure_class"),
        "timed_out": report.get("timed_out"),
        "return_code": report.get("return_code"),
        "log_path": report.get("log_path") or report.get("log_file"),
        "report_dir": report.get("report_dir"),
        "candidate_hash": report.get("candidate_hash"),
    }


def _apply_to_state(
    state: dict[str, Any],
    updated: dict[str, dict[str, Any]],
) -> None:
    def replace(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        source = _source(value)
        if source is None:
            return value
        replacement = updated.get(_identity(value, source))
        if replacement is None:
            return value
        merged = dict(value)
        for key in (
            "candidate_hash",
            "cosim_required",
            "cosim",
            "cosim_run",
            "fully_verified",
            "validation",
            "final_cosim",
        ):
            if key in replacement:
                merged[key] = replacement[key]
        return merged

    for key in (
        "selected_design",
        "latest_candidate",
        "best_correct_candidate",
        "best_ppa_candidate",
        "original_baseline",
    ):
        state[key] = replace(state.get(key))
    state["pareto_archive"] = [replace(item) for item in state.get("pareto_archive", [])]


def _selection_config(state: dict[str, Any]) -> dict[str, Any]:
    policy = state.get("selection_policy")
    if not isinstance(policy, dict):
        return {}
    ranking = policy.get("ranking")
    return {"ranking": list(ranking)} if isinstance(ranking, list) and ranking else {}


def _eligible(record: dict[str, Any]) -> bool:
    return bool(
        record.get("fully_verified") is True
        and record.get("meets_frequency_requirement") is not False
        and record.get("meets_resource_limits") is not False
    )


def _is_baseline(record: dict[str, Any]) -> bool:
    role = str(record.get("role") or "")
    return bool(
        role in {
            "verified_baseline_fallback",
            "original_baseline",
            "baseline",
        }
        or record.get("candidate_index") == 0
    )


def _ranked_validation_order(
    records: list[dict[str, Any]],
    pareto_identities: set[str],
    selected_identity: str | None,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    by_identity = {
        str(record.get("candidate_hash")): record
        for record in records
        if isinstance(record.get("candidate_hash"), str)
    }
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(record: dict[str, Any] | None) -> None:
        if record is None:
            return
        identity = str(record.get("candidate_hash") or "")
        if not identity or identity in seen:
            return
        ordered.append(record)
        seen.add(identity)

    add(by_identity.get(selected_identity or ""))

    pareto = [
        record
        for record in records
        if record.get("candidate_hash") in pareto_identities
        and record.get("candidate_hash") != selected_identity
        and record.get("meets_frequency_requirement") is not False
        and record.get("meets_resource_limits") is not False
    ]
    pareto.sort(
        key=lambda item: deterministic_selection_key(
            item,
            _selection_config(state),
        )
    )
    for record in pareto:
        add(record)

    for record in records:
        if _is_baseline(record):
            add(record)
    return ordered


def _archive_selected(output_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    source = _source(record)
    if source is None:
        raise FileNotFoundError("Selected fallback source is unavailable")
    destination = output_dir / "candidate_archive" / f"selected_design{source.suffix or '.cpp'}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    selected = dict(record)
    selected["role"] = "selected_design"
    selected["candidate_file"] = _display(source)
    selected["archived_file"] = _display(destination)
    selected["candidate_hash"] = _hash(destination)
    return selected


def _select_event(result: AgentResult) -> TrajectoryEvent:
    for event in reversed(result.trajectory):
        if event.stage == "select_best":
            return event
    event = TrajectoryEvent(
        step=len(result.trajectory) + 1,
        stage="select_best",
        status="failed",
        details={},
    )
    result.trajectory.append(event)
    return event


def _cosim_report(
    task: TaskManifest,
    record: dict[str, Any],
    source: Path,
    identity: str,
    cosim_runner: CosimRunner,
) -> tuple[dict[str, Any], bool]:
    validation = _validation(record)
    existing_cosim = validation.get("cosim")
    reused = existing_cosim in {True, False}

    if validation.get("csim") is not True or validation.get("synthesis") is not True:
        return (
            {
                "passed": False,
                "failure_class": "pre_cosim_validation_incomplete",
                "timed_out": False,
                "return_code": None,
                "candidate_hash": identity,
                "candidate_file": _display(source),
            },
            True,
        )
    if reused:
        return (
            {
                "passed": existing_cosim is True,
                "failure_class": (
                    "none" if existing_cosim is True else "previous_cosim_failure"
                ),
                "timed_out": False,
                "return_code": 0 if existing_cosim is True else None,
                "candidate_hash": identity,
                "candidate_file": _display(source),
            },
            True,
        )
    try:
        return cosim_runner(task, source), False
    except Exception as error:
        return (
            {
                "passed": False,
                "failure_class": type(error).__name__,
                "timed_out": False,
                "return_code": None,
                "candidate_hash": identity,
                "candidate_file": _display(source),
                "evidence": [str(error)],
            },
            False,
        )


def enforce_final_cosim_policy(
    task: TaskManifest,
    result: AgentResult,
    *,
    cosim_runner: CosimRunner = run_cosim,
) -> dict[str, Any]:
    """Validate the selected design, then ranked fallbacks, stopping on success."""

    output_dir = _resolve(task.output_dir)
    state_path = output_dir / "candidate_state.json"
    state = _load_json(state_path)
    audit_path = output_dir / "final_cosim_audit.json"

    if not state:
        audit = {
            "schema_version": 2,
            "policy": POLICY_NAME,
            "status": "skipped",
            "reason": "no_selectable_state",
            "metered_agent_budget": False,
            "candidates": [],
        }
        _write_json(audit_path, audit)
        return audit

    records, pareto_identities, selected_identity = _collect_records(state, output_dir)
    order = _ranked_validation_order(
        records,
        pareto_identities,
        selected_identity,
        state,
    )
    if not order:
        audit = {
            "schema_version": 2,
            "policy": POLICY_NAME,
            "status": "failed",
            "reason": "no_candidate_sources",
            "metered_agent_budget": False,
            "candidates": [],
        }
        _write_json(audit_path, audit)
        result.success = False
        result.status = "final_cosim_failed"
        result.termination_reason = "no_candidate_available_for_final_cosim"
        return audit

    updated: dict[str, dict[str, Any]] = {}
    audited: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None

    for rank, record in enumerate(order, 1):
        source = _source(record)
        if source is None:
            continue
        identity = _identity(record, source)
        report, reused = _cosim_report(
            task,
            record,
            source,
            identity,
            cosim_runner,
        )
        current = dict(record)
        _update_record(current, report)
        updated[identity] = current
        audited.append(
            {
                "validation_rank": rank,
                "candidate_hash": identity,
                "candidate_index": current.get("candidate_index"),
                "role": current.get("role"),
                "source": _display(source),
                "was_selected": identity == selected_identity,
                "was_pareto": identity in pareto_identities,
                "reused_existing_cosim": reused,
                "passed": _eligible(current),
                "failure_class": report.get("failure_class"),
                "timed_out": report.get("timed_out"),
                "log_path": report.get("log_path") or report.get("log_file"),
            }
        )
        if _eligible(current):
            chosen = current
            break

    _apply_to_state(state, updated)
    verified_pareto = [
        record
        for identity, record in updated.items()
        if identity in pareto_identities and _eligible(record)
    ]
    verified_pareto.sort(
        key=lambda item: deterministic_selection_key(
            item,
            _selection_config(state),
        )
    )
    state["pareto_archive"] = verified_pareto
    fallback_used = bool(
        chosen is not None
        and str(chosen.get("candidate_hash")) != (selected_identity or "")
    )
    select_event = _select_event(result)

    if chosen is None:
        state["selected_design"] = None
        state["selected_design_fully_verified"] = False
        state["selected_design_frequency_compliant"] = False
        state["selected_design_resource_compliant"] = False
        state["best_ppa_candidate"] = None
        state["best_correct_candidate"] = None
        result.success = False
        result.status = "final_cosim_failed"
        result.termination_reason = "no_cosim_verified_pareto_or_baseline"
        select_event.status = "failed"
        select_event.details.update(
            {
                "selected_design": None,
                "selected_design_fully_verified": False,
                "selected_design_frequency_compliant": False,
                "selected_design_resource_compliant": False,
                "best_ppa_candidate": None,
                "best_correct_candidate": None,
                "pareto_archive": [],
                "final_cosim_audit": _display(audit_path),
                "fallback_used": bool(selected_identity),
            }
        )
        status = "failed"
    else:
        selected = _archive_selected(output_dir, chosen)
        chosen_identity = str(chosen.get("candidate_hash"))
        state["selected_design"] = selected
        state["best_correct_candidate"] = selected
        state["selected_design_fully_verified"] = True
        state["selected_design_frequency_compliant"] = selected.get(
            "meets_frequency_requirement"
        ) is not False
        state["selected_design_resource_compliant"] = selected.get(
            "meets_resource_limits"
        ) is not False
        if chosen_identity in pareto_identities:
            state["best_ppa_candidate"] = selected
        else:
            state["best_ppa_candidate"] = None
        if fallback_used and result.success:
            result.status = "completed_with_cosim_fallback"
            result.termination_reason = "selected_cosim_failed_fallback_verified"
        select_event.status = "passed"
        select_event.details.update(
            {
                "selected_design": selected["archived_file"],
                "selected_design_fully_verified": True,
                "selected_design_frequency_compliant": state[
                    "selected_design_frequency_compliant"
                ],
                "selected_design_resource_compliant": state[
                    "selected_design_resource_compliant"
                ],
                "best_ppa_candidate": (
                    selected["archived_file"]
                    if chosen_identity in pareto_identities
                    else None
                ),
                "best_correct_candidate": selected["archived_file"],
                "pareto_archive": [
                    item.get("archived_file") for item in verified_pareto
                ],
                "final_cosim_audit": _display(audit_path),
                "fallback_used": fallback_used,
                "selected_before_cosim": selected_identity,
                "selected_after_cosim": chosen_identity,
            }
        )
        status = "passed_with_fallback" if fallback_used else "passed"

    policy = state.get("selection_policy")
    policy = dict(policy) if isinstance(policy, dict) else {}
    policy.update(
        {
            "final_cosim_required": True,
            "pareto_cosim_required": False,
            "final_cosim_strategy": POLICY_NAME,
            "fallback_cosim_required": True,
            "post_search_audit_unmetered": True,
        }
    )
    state["selection_policy"] = policy
    _write_json(state_path, state)

    audit = {
        "schema_version": 2,
        "policy": POLICY_NAME,
        "status": status,
        "metered_agent_budget": False,
        "budget_scope": (
            "Post-search validation audit. Search-time tool credits remain unchanged."
        ),
        "selected_before_cosim": selected_identity,
        "selected_after_cosim": (
            str(chosen.get("candidate_hash")) if chosen is not None else None
        ),
        "fallback_used": fallback_used,
        "validation_order_size": len(order),
        "candidates_audited": len(audited),
        "candidates_skipped_after_success": max(len(order) - len(audited), 0),
        "verified_pareto_count": len(verified_pareto),
        "candidates": audited,
    }
    _write_json(audit_path, audit)

    result.trajectory.append(
        TrajectoryEvent(
            step=len(result.trajectory) + 1,
            stage="final_cosim_audit",
            status="passed" if chosen is not None else "failed",
            details={
                "audit": _display(audit_path),
                "policy": POLICY_NAME,
                "candidates_audited": len(audited),
                "candidates_skipped_after_success": audit[
                    "candidates_skipped_after_success"
                ],
                "verified_pareto_count": len(verified_pareto),
                "fallback_used": fallback_used,
                "selected_after_cosim": audit["selected_after_cosim"],
                "metered_agent_budget": False,
            },
        )
    )
    _write_json(output_dir / "unified_agent_result.json", result.to_dict())
    return audit
