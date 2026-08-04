"""Deterministic refinement-parent selection for PPA candidates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _parent_rank(record: dict[str, Any]) -> tuple[int, int, str] | None:
    index = record.get("candidate_index")
    if not isinstance(index, int):
        return None
    if record.get("verdict") == "reject_duplicate":
        return None
    if record.get("fully_verified") is True:
        return 4, index, "fully_verified_candidate"
    if record.get("synthesis") is True and record.get("refinement_eligible") is True:
        return 3, index, "synthesis_passed_refinement_eligible"
    if record.get("csim") is True:
        return 2, index, "csim_passed_candidate"
    if record.get("static_validation") is True:
        return 1, index, "static_valid_candidate"
    return 0, index, "latest_non_duplicate_fallback"


def select_refinement_parent(
    records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], str] | None:
    """Return the strongest parent, preferring the latest within a quality tier."""
    ranked = [
        (rank, record)
        for record in records
        if (rank := _parent_rank(record)) is not None
    ]
    if not ranked:
        return None
    rank, record = max(ranked, key=lambda item: item[0][:2])
    return record, rank[2]
