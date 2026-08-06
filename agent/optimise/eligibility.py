"""Separate archival, refinement and final-selection eligibility.

These predicates deliberately keep three different questions independent:

* archive_eligible: is the candidate a valid non-dominated research result?
* refinement_eligible: is it useful enough to spend another model/tool budget on?
* final_selectable: is it currently safe and eligible for final ranking?

The module is pure and does not perform file I/O.  Evaluation integration is kept
separate so this policy can be regression-tested before changing the search loop.
"""

from __future__ import annotations

from typing import Any

ARCHIVE_ELIGIBLE_VERDICTS = {
    "accept_dominates_baseline",
    "keep_pareto_candidate",
}
RECOVERABLE_CONSTRAINT_VERDICTS = {
    "reject_frequency_threshold",
    "reject_resource_limits",
}

MIN_PERFORMANCE_IMPROVEMENT_PERCENT = 5.0
MAX_LATENCY_REGRESSION_FOR_AREA_TRADEOFF_PERCENT = 10.0
MIN_AREA_REDUCTION_PERCENT = 15.0
AREA_METRICS = (
    "resources_lut_used",
    "resources_ff_used",
)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _constraint_passed(record: dict[str, Any]) -> bool:
    compliance = record.get("resource_limit_compliance")
    if isinstance(compliance, dict):
        return compliance.get("passed") is True
    return record.get("meets_resource_limits") is True


def _validation_complete(record: dict[str, Any]) -> bool:
    requires_cosim = bool(record.get("cosim_required", True))
    return bool(
        record.get("fully_verified") is True
        and record.get("static_validation") is True
        and record.get("csim") is True
        and record.get("synthesis") is True
        and (record.get("cosim") is True if requires_cosim else True)
    )


def _performance_deltas(record: dict[str, Any]) -> tuple[float | None, float | None]:
    comparison = record.get("performance_comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    deltas = record.get("deltas_percent")
    deltas = deltas if isinstance(deltas, dict) else {}

    latency = _number(comparison.get("latency_delta_percent"))
    if latency is None:
        latency = _number(deltas.get("latency_ns"))

    throughput = _number(comparison.get("throughput_delta_percent"))
    if throughput is None:
        throughput = _number(deltas.get("throughput_period_ns"))
    return latency, throughput


def practical_refinement_signal(record: dict[str, Any]) -> bool:
    """Return whether measured evidence justifies another search iteration.

    A candidate is practically useful when it provides a material performance
    improvement, or when latency remains close to the baseline while LUT or FF
    usage falls materially.  A DSP-only reduction cannot justify an extreme
    latency regression.
    """

    latency_delta, throughput_delta = _performance_deltas(record)
    performance_deltas = [
        value
        for value in (latency_delta, throughput_delta)
        if value is not None
    ]
    material_performance_gain = any(
        value <= -MIN_PERFORMANCE_IMPROVEMENT_PERCENT
        for value in performance_deltas
    )
    if material_performance_gain:
        return True

    # Require every available performance measure to remain within the allowed
    # regression envelope.  Missing performance evidence is not sufficient.
    if not performance_deltas or any(
        value > MAX_LATENCY_REGRESSION_FOR_AREA_TRADEOFF_PERCENT
        for value in performance_deltas
    ):
        return False

    deltas = record.get("deltas_percent")
    deltas = deltas if isinstance(deltas, dict) else {}
    return any(
        (value := _number(deltas.get(metric))) is not None
        and value <= -MIN_AREA_REDUCTION_PERCENT
        for metric in AREA_METRICS
    )


def candidate_eligibility(record: dict[str, Any]) -> dict[str, bool]:
    """Return explicit archive, refinement and final-selection flags."""

    verdict = record.get("verdict")
    frequency_ok = record.get("meets_frequency_requirement") is True
    resources_ok = _constraint_passed(record)
    validation_ok = _validation_complete(record)

    if verdict == "baseline":
        baseline_ok = validation_ok and frequency_ok and resources_ok
        return {
            "archive_eligible": baseline_ok,
            "refinement_eligible": baseline_ok,
            "final_selectable": baseline_ok,
        }

    archive_eligible = bool(
        verdict in ARCHIVE_ELIGIBLE_VERDICTS
        and validation_ok
        and frequency_ok
        and resources_ok
    )

    recoverable_constraint = bool(
        verdict in RECOVERABLE_CONSTRAINT_VERDICTS
        and record.get("static_validation") is True
        and record.get("csim") is True
        and record.get("synthesis") is True
    )

    refinement_eligible = bool(
        (archive_eligible or recoverable_constraint)
        and practical_refinement_signal(record)
    )

    final_selectable = bool(archive_eligible and validation_ok)

    return {
        "archive_eligible": archive_eligible,
        "refinement_eligible": refinement_eligible,
        "final_selectable": final_selectable,
    }


def annotate_candidate_eligibility(record: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy containing the explicit eligibility fields."""

    annotated = dict(record)
    annotated.update(candidate_eligibility(record))
    return annotated
