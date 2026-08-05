"""Derived hardware timing metrics and frequency-constraint helpers."""

from __future__ import annotations

from typing import Any, Mapping


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _positive(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _cycles_times_period(cycles: Any, period_ns: float | None) -> float | None:
    cycle_count = _number(cycles)
    if cycle_count is None or cycle_count < 0 or period_ns is None:
        return None
    return cycle_count * period_ns


def maximum_clock_period_ns(minimum_frequency_mhz: Any) -> float | None:
    """Return the longest period that still satisfies a frequency requirement."""
    frequency = _positive(minimum_frequency_mhz)
    return None if frequency is None else 1000.0 / frequency


def derive_hardware_metrics(
    metrics: Mapping[str, Any],
    *,
    minimum_frequency_mhz: Any = None,
) -> dict[str, Any]:
    """Preserve synthesis metrics and add real-time/frequency measurements.

    Canonical FPT-602 fields are based on the best latency and minimum
    initiation interval. Additional best/average/worst aliases are retained for
    analysis without changing existing cycle-based fields.
    """
    result = dict(metrics)
    period = _positive(result.get("clock_period_ns"))
    frequency = None if period is None else 1000.0 / period

    latency_best = _cycles_times_period(result.get("latency_best_cycles"), period)
    latency_average = _cycles_times_period(result.get("latency_average_cycles"), period)
    latency_worst = _cycles_times_period(result.get("latency_worst_cycles"), period)
    interval_min = _cycles_times_period(result.get("interval_min_cycles"), period)
    interval_max = _cycles_times_period(result.get("interval_max_cycles"), period)

    result.update(
        {
            "frequency_mhz": frequency,
            "latency_ns": latency_best,
            "latency_best_ns": latency_best,
            "latency_average_ns": latency_average,
            "latency_worst_ns": latency_worst,
            "throughput_period_ns": interval_min,
            "throughput_period_min_ns": interval_min,
            "throughput_period_max_ns": interval_max,
        }
    )

    minimum = _positive(minimum_frequency_mhz)
    limit = maximum_clock_period_ns(minimum)
    result["minimum_frequency_mhz"] = minimum
    result["maximum_clock_period_ns"] = limit
    result["meets_minimum_frequency"] = (
        None if minimum is None or frequency is None else frequency >= minimum
    )
    return result


def comparison_metric(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    actual_key: str,
    cycle_key: str,
) -> tuple[str, float | None, float | None]:
    """Prefer actual time, retaining cycle fallback for legacy partial records."""
    candidate_actual = _number(candidate.get(actual_key))
    baseline_actual = _number(baseline.get(actual_key))
    if candidate_actual is not None and baseline_actual is not None:
        return actual_key, candidate_actual, baseline_actual
    return cycle_key, _number(candidate.get(cycle_key)), _number(baseline.get(cycle_key))


def metric_delta_percent(candidate: Any, baseline: Any) -> float | None:
    candidate_value = _number(candidate)
    baseline_value = _number(baseline)
    if candidate_value is None or baseline_value is None:
        return None
    if baseline_value == 0:
        return 0.0 if candidate_value == 0 else None
    return ((candidate_value - baseline_value) / baseline_value) * 100.0
