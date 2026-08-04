"""Deterministic refinement-parent selection for PPA candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent.optimise.refinement_strategy import LATENCY_RECOVERY_FACTORS
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


def _is_latency_recovery_opportunity(record: dict[str, Any]) -> bool:
    if not (
        record.get("fully_verified") is True
        and record.get("verdict") == "keep_pareto_candidate"
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

    for record in records:
        if not _is_latency_recovery_opportunity(record):
            continue
        index = record.get("candidate_index")
        output_dir = _candidate_output_dir(record)
        if not isinstance(index, int) or output_dir is None:
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


def _parent_rank(record: dict[str, Any]) -> tuple[int, int, str] | None:
    index = record.get("candidate_index")
    if not isinstance(index, int):
        return None

    verdict = record.get("verdict")
    if verdict == "reject_duplicate":
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


def select_refinement_parent(
    records: Iterable[dict[str, Any]],
    selection: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str] | None:
    """Return a pending bounded strategy parent, then the strongest normal parent."""
    record_list = list(records)

    pending = _pending_latency_recovery_parent(record_list, selection)
    if pending is not None:
        return pending

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
