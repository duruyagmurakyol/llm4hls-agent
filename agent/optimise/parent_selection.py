"""Deterministic refinement-parent selection for PPA candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent.optimise.refinement_strategy import LATENCY_RECOVERY_FACTORS
from agent.optimise.resource_recovery import (
    RESOURCE_FREQUENCY_BALANCE_REASON,
    resource_frequency_balance_trigger,
    resource_limit_recovery_trigger,
)
from agent.optimise.selection import deterministic_selection_key

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_KEYS = (
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
)
LATENCY_RECOVERY_THRESHOLD_PERCENT = 50.0
RESOURCE_RECOVERY_THRESHOLD_PERCENT = 25.0

BASELINE_RESTART_REASON = "restart_from_verified_baseline"
BASELINE_RESTART_VERDICTS = {
    "reject_duplicate",
    "reject_no_change",
    "reject_no_objective_gain",
    "reject_dominated_pre_cosim",
    "reject_no_change_pre_cosim",
    "reject_synthesis_equivalent",
}


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _candidate_output_dir(record: dict[str, Any]) -> Path | None:
    value = record.get("candidate_file")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.parent


def _candidate_strategy(record: dict[str, Any]) -> dict[str, Any]:
    index = record.get("candidate_index")
    output_dir = _candidate_output_dir(record)
    if not isinstance(index, int) or output_dir is None:
        return {}
    return _load_optional_json(
        output_dir / f"candidate_{index:03d}_strategy.json"
    )


def _latency_recovery_source(record: dict[str, Any]) -> int | None:
    strategy = _candidate_strategy(record)
    if strategy.get("name") != "recover_latency_tradeoff":
        return None
    source = strategy.get("source_candidate_index")
    return source if isinstance(source, int) else None


def _is_latency_recovery_descendant(record: dict[str, Any]) -> bool:
    index = record.get("candidate_index")
    source = _latency_recovery_source(record)
    return isinstance(index, int) and isinstance(source, int) and source != index


def _failed_latency_recovery_sources(
    records: list[dict[str, Any]],
) -> set[int]:
    failed: set[int] = set()
    for record in records:
        source = _latency_recovery_source(record)
        index = record.get("candidate_index")
        output_dir = _candidate_output_dir(record)
        if (
            not isinstance(source, int)
            or not isinstance(index, int)
            or output_dir is None
        ):
            continue

        if record.get("verdict") == "reject_strategy_not_realised":
            failed.add(source)
            continue

        if record.get("verdict") != "reject_static":
            continue
        static = _load_optional_json(
            output_dir / f"candidate_{index:03d}_static_validation.json"
        )
        compliance = static.get("strategy_compliance")
        if (
            isinstance(compliance, dict)
            and compliance.get("required") is True
            and compliance.get("passed") is False
        ):
            failed.add(source)
    return failed


def _is_latency_recovery_opportunity(record: dict[str, Any]) -> bool:
    if not (
        record.get("fully_verified") is True
        and record.get("verdict") == "keep_pareto_candidate"
        and record.get("refinement_eligible") is not False
    ):
        return False

    deltas = record.get("deltas_percent")
    if not isinstance(deltas, dict):
        return False

    latency_regression = deltas.get("latency_ns")
    if (
        isinstance(latency_regression, bool)
        or not isinstance(latency_regression, (int, float))
        or latency_regression < LATENCY_RECOVERY_THRESHOLD_PERCENT
    ):
        return False

    return any(
        isinstance((reduction := deltas.get(key)), (int, float))
        and not isinstance(reduction, bool)
        and reduction <= -RESOURCE_RECOVERY_THRESHOLD_PERCENT
        for key in RESOURCE_KEYS
    )


def _completed_latency_recovery_factors(
    output_dir: Path,
    source_candidate_index: int,
) -> set[int]:
    completed: set[int] = set()

    for path in output_dir.glob("candidate_*_strategy.json"):
        strategy = _load_optional_json(path)
        if not (
            strategy.get("name") == "recover_latency_tradeoff"
            and strategy.get("source_candidate_index") == source_candidate_index
        ):
            continue
        factor = (strategy.get("parameters") or {}).get("factor")
        if isinstance(factor, int) and factor in LATENCY_RECOVERY_FACTORS:
            completed.add(factor)

    for path in output_dir.glob("candidate_*_strategy_exhausted.json"):
        strategy = _load_optional_json(path)
        if not (
            strategy.get("name") == "recover_latency_tradeoff"
            and strategy.get("source_candidate_index") == source_candidate_index
        ):
            continue
        for factor in strategy.get("completed_factors") or []:
            if isinstance(factor, int) and factor in LATENCY_RECOVERY_FACTORS:
                completed.add(factor)
        if strategy.get("status") == "exhausted":
            completed.update(LATENCY_RECOVERY_FACTORS)

    return completed


def _pending_latency_recovery_parent(
    records: list[dict[str, Any]],
    selection: dict[str, Any] | None,
) -> tuple[dict[str, Any], str] | None:
    pending: list[dict[str, Any]] = []
    required_factors = set(LATENCY_RECOVERY_FACTORS)
    failed_sources = _failed_latency_recovery_sources(records)

    for record in records:
        if not _is_latency_recovery_opportunity(record):
            continue
        if _is_latency_recovery_descendant(record):
            continue
        index = record.get("candidate_index")
        output_dir = _candidate_output_dir(record)
        if not isinstance(index, int) or output_dir is None:
            continue
        if index in failed_sources:
            continue
        completed = _completed_latency_recovery_factors(output_dir, index)
        if not required_factors.issubset(completed):
            pending.append(record)

    if not pending:
        return None

    record = min(
        pending,
        key=lambda item: deterministic_selection_key(item, selection),
    )
    return record, "pending_latency_recovery_strategy"


def _is_feasible_verified_parent(record: dict[str, Any]) -> bool:
    compliance = record.get("resource_limit_compliance")
    return bool(
        record.get("fully_verified") is True
        and record.get("meets_frequency_requirement") is True
        and isinstance(compliance, dict)
        and compliance.get("passed") is True
        and record.get("verdict")
        not in {
            "reject_no_objective_gain",
            "reject_dominated_pre_cosim",
            "reject_no_change_pre_cosim",
            "reject_resource_limits",
        }
    )


def _pending_resource_frequency_balance_parent(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], str] | None:
    """Return to the original feasible parent after recovery breaks timing."""
    trigger = resource_frequency_balance_trigger(records)
    if trigger is None:
        return None
    parent = trigger.get("parent")
    if not isinstance(parent, dict):
        return None
    return parent, RESOURCE_FREQUENCY_BALANCE_REASON


def _pending_resource_limit_recovery_parent(
    records: list[dict[str, Any]],
    selection: dict[str, Any] | None,
) -> tuple[dict[str, Any], str] | None:
    """Return to the best feasible design after an over-budget attempt."""
    if resource_limit_recovery_trigger(records) is None:
        return None

    # A fully verified no-change or synthesis-equivalent candidate is still a
    # safe recovery anchor: it is effectively the baseline, but unlike the
    # rejected over-budget child it has a concrete candidate source file.
    feasible = [record for record in records if _is_feasible_verified_parent(record)]
    if not feasible:
        return None

    pareto = [
        record
        for record in feasible
        if record.get("verdict") in {
            "accept_dominates_baseline",
            "keep_pareto_candidate",
        }
        and record.get("refinement_eligible") is not False
    ]
    candidates = pareto or feasible
    record = min(
        candidates,
        key=lambda item: deterministic_selection_key(item, selection),
    )
    reason = (
        "resource_limit_recovery_from_feasible_pareto"
        if pareto
        else "resource_limit_recovery_from_feasible_verified"
    )
    return record, reason


def _parent_rank(record: dict[str, Any]) -> tuple[int, int, str] | None:
    index = record.get("candidate_index")
    if not isinstance(index, int):
        return None

    # Stage-3 summaries explicitly separate archive membership from whether a
    # candidate is useful enough to spend another search iteration on. Legacy
    # records without this field retain their previous behaviour.
    if record.get("refinement_eligible") is False:
        return None

    verdict = record.get("verdict")
    if verdict in {
        "reject_duplicate",
        "reject_no_change",
        "reject_no_objective_gain",
        "reject_dominated_pre_cosim",
        "reject_no_change_pre_cosim",
        "reject_resource_limits",
        "reject_synthesis_equivalent",
    }:
        return None

    fully_verified = record.get("fully_verified") is True
    if fully_verified and verdict == "accept_dominates_baseline":
        return 6, index, "dominates_baseline_candidate"
    if fully_verified and verdict == "keep_pareto_candidate":
        return 5, index, "pareto_candidate"
    if record.get("synthesis") is True and record.get("refinement_eligible") is True:
        return 4, index, "synthesis_passed_refinement_eligible"
    if fully_verified:
        return 3, index, "fully_verified_candidate"
    if record.get("csim") is True:
        return 2, index, "csim_passed_candidate"
    if record.get("static_validation") is True:
        return 1, index, "static_valid_candidate"
    return 0, index, "latest_non_duplicate_fallback"


def _baseline_restart_parent(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], str] | None:
    """Restart from baseline after an objectively useless or retired result."""

    indexed = [
        record
        for record in records
        if isinstance(record.get("candidate_index"), int)
    ]
    if not indexed:
        return None

    latest = max(indexed, key=lambda item: int(item["candidate_index"]))
    explicitly_retired = bool(
        "refinement_eligible" in latest
        and latest.get("refinement_eligible") is False
    )
    if (
        latest.get("verdict") not in BASELINE_RESTART_VERDICTS
        and not explicitly_retired
    ):
        return None

    # A genuinely useful Pareto point, dominating candidate, or explicitly
    # recoverable constraint violation remains a better parent than baseline.
    for record in records:
        rank = _parent_rank(record)
        if rank is not None and rank[0] >= 4:
            return None

    trigger_index = int(latest["candidate_index"])
    baseline = {
        "candidate_index": 0,
        "candidate_file": "verified_baseline",
        "fully_verified": True,
        "verdict": "baseline_restart",
        "trigger_candidate_index": trigger_index,
        "trigger_verdict": latest.get("verdict"),
    }
    return baseline, BASELINE_RESTART_REASON


def select_refinement_parent(
    records: Iterable[dict[str, Any]],
    selection: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str] | None:
    """Return a pending recovery parent, then the strongest normal parent."""
    record_list = list(records)

    balanced_recovery = _pending_resource_frequency_balance_parent(record_list)
    if balanced_recovery is not None:
        return balanced_recovery

    resource_recovery = _pending_resource_limit_recovery_parent(
        record_list,
        selection,
    )
    if resource_recovery is not None:
        return resource_recovery

    pending = _pending_latency_recovery_parent(record_list, selection)
    if pending is not None:
        return pending

    baseline_restart = _baseline_restart_parent(record_list)
    if baseline_restart is not None:
        return baseline_restart

    ranked = [
        (rank, record)
        for record in record_list
        if (rank := _parent_rank(record)) is not None
    ]
    if not ranked:
        return None

    strongest_tier = max(rank[0] for rank, _ in ranked)
    strongest = [
        (rank, record)
        for rank, record in ranked
        if rank[0] == strongest_tier
    ]

    if strongest_tier in {4, 5, 6}:
        rank, record = min(
            strongest,
            key=lambda item: deterministic_selection_key(item[1], selection),
        )
    else:
        rank, record = max(strongest, key=lambda item: item[0][1])
    return record, rank[2]
