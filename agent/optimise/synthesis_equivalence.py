"""Detect fully verified candidates that synthesise to the same hardware result."""

from __future__ import annotations

import math
from typing import Any

SYNTHESIS_EQUIVALENCE_REL_TOLERANCE = 1e-6
SYNTHESIS_EQUIVALENCE_ABS_TIME_TOLERANCE_NS = 1e-3

RESOURCE_KEYS = (
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
)
PRIMARY_CYCLE_KEYS = (
    "latency_best_cycles",
    "interval_min_cycles",
)
OPTIONAL_CYCLE_KEYS = (
    "latency_average_cycles",
    "latency_worst_cycles",
    "interval_max_cycles",
)
FALLBACK_TIME_KEYS = (
    "latency_ns",
    "throughput_period_ns",
)
PARETO_ELIGIBLE_VERDICTS = {
    "accept_dominates_baseline",
    "keep_pareto_candidate",
}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _close_time(left: Any, right: Any) -> bool:
    return bool(
        _number(left)
        and _number(right)
        and math.isclose(
            float(left),
            float(right),
            rel_tol=SYNTHESIS_EQUIVALENCE_REL_TOLERANCE,
            abs_tol=SYNTHESIS_EQUIVALENCE_ABS_TIME_TOLERANCE_NS,
        )
    )


def synthesis_equivalence_evidence(
    left_metrics: dict[str, Any],
    right_metrics: dict[str, Any],
) -> dict[str, Any] | None:
    """Return conservative equivalence evidence or ``None``.

    Resource counts and cycle counts must match exactly. Floating-point timing
    values may differ only within report-rounding tolerance. A genuine one-cycle
    or one-resource change is therefore preserved as a distinct hardware result.
    """

    matched: list[str] = []
    for key in RESOURCE_KEYS:
        left = left_metrics.get(key)
        right = right_metrics.get(key)
        if not (_number(left) and _number(right) and float(left) == float(right)):
            return None
        matched.append(key)

    left_clock = left_metrics.get("clock_period_ns")
    right_clock = right_metrics.get("clock_period_ns")
    if not _close_time(left_clock, right_clock):
        return None
    matched.append("clock_period_ns")

    has_cycle_basis = all(
        _number(left_metrics.get(key)) and _number(right_metrics.get(key))
        for key in PRIMARY_CYCLE_KEYS
    )
    if has_cycle_basis:
        for key in PRIMARY_CYCLE_KEYS:
            if float(left_metrics[key]) != float(right_metrics[key]):
                return None
            matched.append(key)

        for key in OPTIONAL_CYCLE_KEYS:
            left = left_metrics.get(key)
            right = right_metrics.get(key)
            if left is None and right is None:
                continue
            if not (_number(left) and _number(right) and float(left) == float(right)):
                return None
            matched.append(key)
        timing_basis = "cycles_and_clock"
    else:
        for key in FALLBACK_TIME_KEYS:
            if not _close_time(left_metrics.get(key), right_metrics.get(key)):
                return None
            matched.append(key)
        timing_basis = "derived_time"

    return {
        "timing_basis": timing_basis,
        "matched_metrics": matched,
        "relative_tolerance": SYNTHESIS_EQUIVALENCE_REL_TOLERANCE,
        "absolute_time_tolerance_ns": (
            SYNTHESIS_EQUIVALENCE_ABS_TIME_TOLERANCE_NS
        ),
        "exact_cycle_and_resource_counts": True,
    }


def _verified_and_compliant(record: dict[str, Any]) -> bool:
    compliance = record.get("resource_limit_compliance")
    return bool(
        record.get("fully_verified") is True
        and record.get("meets_frequency_requirement") is True
        and isinstance(compliance, dict)
        and compliance.get("passed") is True
        and isinstance(record.get("metrics"), dict)
    )


def apply_synthesis_equivalence(summary: dict[str, Any]) -> dict[str, Any]:
    """Reject later hardware-equivalent candidates and remove them from Pareto.

    The earliest fully verified, timing-compliant and resource-compliant design
    is retained as the representative. Source-level differences alone are not
    enough to create a new Pareto point.
    """

    result = dict(summary)
    candidates = [
        dict(item)
        for item in summary.get("candidates", [])
        if isinstance(item, dict)
    ]
    result["candidates"] = candidates

    references: list[dict[str, Any]] = []
    baseline = summary.get("baseline_record")
    if isinstance(baseline, dict) and _verified_and_compliant(baseline):
        references.append(dict(baseline))

    equivalent_candidates: list[dict[str, Any]] = []
    for record in sorted(
        candidates,
        key=lambda item: int(item.get("candidate_index", -1)),
    ):
        if not _verified_and_compliant(record):
            continue

        evidence: dict[str, Any] | None = None
        representative: dict[str, Any] | None = None
        for earlier in references:
            evidence = synthesis_equivalence_evidence(
                record.get("metrics") or {},
                earlier.get("metrics") or {},
            )
            if evidence is not None:
                representative = earlier
                break

        if representative is None or evidence is None:
            references.append(record)
            continue

        representative_index = int(representative.get("candidate_index", 0))
        record["synthesis_equivalent_to"] = representative_index
        record["synthesis_equivalence"] = evidence
        record["refinement_eligible"] = False
        record["pareto"] = False

        if representative_index == 0:
            record["verdict"] = "reject_no_change"
            record["reason"] = (
                "Synthesis metrics are equivalent to the verified baseline "
                "within the configured report-rounding tolerance."
            )
        else:
            record["verdict"] = "reject_synthesis_equivalent"
            record["reason"] = (
                "Synthesis metrics are equivalent to "
                f"candidate_{representative_index:03d} within the configured "
                "report-rounding tolerance."
            )

        equivalent_candidates.append(
            {
                "candidate_index": record.get("candidate_index"),
                "synthesis_equivalent_to": representative_index,
                "timing_basis": evidence["timing_basis"],
            }
        )

    current_by_index = {
        item.get("candidate_index"): item
        for item in candidates
        if isinstance(item.get("candidate_index"), int)
    }
    filtered_archive: list[dict[str, Any]] = []
    for item in summary.get("pareto_archive", []):
        if not isinstance(item, dict):
            continue
        index = item.get("candidate_index")
        if index == 0:
            filtered_archive.append(item)
            continue
        current = current_by_index.get(index)
        if (
            isinstance(current, dict)
            and current.get("verdict") in PARETO_ELIGIBLE_VERDICTS
        ):
            filtered_archive.append(item)
    result["pareto_archive"] = filtered_archive

    result["schema_version"] = max(int(summary.get("schema_version", 0)), 8)
    result["synthesis_equivalence_policy"] = {
        "description": (
            "Cycle counts and resource counts must match exactly; timing-report "
            "floats may differ only within rounding tolerance."
        ),
        "relative_tolerance": SYNTHESIS_EQUIVALENCE_REL_TOLERANCE,
        "absolute_time_tolerance_ns": (
            SYNTHESIS_EQUIVALENCE_ABS_TIME_TOLERANCE_NS
        ),
        "exact_resource_metrics": list(RESOURCE_KEYS),
        "primary_cycle_metrics": list(PRIMARY_CYCLE_KEYS),
    }
    result["synthesis_equivalent_candidates"] = equivalent_candidates
    return result
