"""Deterministic refinement-parent selection for PPA candidates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent.optimise.selection import deterministic_selection_key


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
    """Return the strongest parent using PPA ranking for synthesis-backed tiers."""
    ranked = [
        (rank, record)
        for record in records
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
