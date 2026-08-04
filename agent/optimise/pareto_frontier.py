"""Materialise current Pareto membership on evaluated candidate records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OBJECTIVES = (
    "latency_ns",
    "throughput_period_ns",
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
)
PARETO_ELIGIBLE_VERDICTS = {
    "accept_dominates_baseline",
    "keep_pareto_candidate",
}


def _objective_values(record: dict[str, Any]) -> tuple[float, ...] | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    values: list[float] = []
    for key in OBJECTIVES:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(float(value))
    return tuple(values)


def record_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether left is no worse in every objective and better in one."""
    left_values = _objective_values(left)
    right_values = _objective_values(right)
    if left_values is None or right_values is None:
        return False
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def annotate_pareto_frontier(
    output_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Annotate candidates and write a standalone current-frontier artefact."""
    frontier = [
        item
        for item in summary.get("pareto_archive", [])
        if isinstance(item, dict)
    ]
    frontier_indices = {
        int(item["candidate_index"])
        for item in frontier
        if isinstance(item.get("candidate_index"), int)
    }

    candidates = [
        item
        for item in summary.get("candidates", [])
        if isinstance(item, dict)
    ]
    dominated: list[dict[str, Any]] = []
    for record in candidates:
        index = record.get("candidate_index")
        is_member = isinstance(index, int) and index in frontier_indices
        record["pareto"] = is_member
        record["dominated_by"] = None

        if record.get("fully_verified") is not True or is_member:
            continue

        dominators = [item for item in frontier if record_dominates(item, record)]
        if not dominators:
            continue

        # Prefer the newest candidate frontier member over the baseline when
        # several designs dominate the same historical candidate.
        dominator = max(
            dominators,
            key=lambda item: int(item.get("candidate_index", 0)),
        )
        dominated_by = int(dominator.get("candidate_index", 0))
        record["dominated_by"] = dominated_by
        if record.get("verdict") in PARETO_ELIGIBLE_VERDICTS:
            label = "baseline" if dominated_by == 0 else f"candidate_{dominated_by:03d}"
            record["verdict"] = "reject_dominated"
            record["reason"] = (
                f"Fully verified candidate is dominated across all Pareto objectives by {label}."
            )
        dominated.append(
            {
                "candidate_index": index,
                "dominated_by": dominated_by,
            }
        )

    payload = {
        "schema_version": 1,
        "objectives": list(OBJECTIVES),
        "members": frontier,
        "dominated_candidates": dominated,
    }
    summary["pareto_frontier"] = payload
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pareto_frontier.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
