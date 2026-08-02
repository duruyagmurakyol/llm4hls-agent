"""Benchmark-independent Pareto comparison for HLS metrics."""

from __future__ import annotations

from agent.state import SynthesisMetrics


def dominates(left: SynthesisMetrics, right: SynthesisMetrics) -> bool:
    left_values = (left.latency_cycles, left.lut, left.ff, left.dsp, left.bram)
    right_values = (right.latency_cycles, right.lut, right.ff, right.dsp, right.bram)
    if any(value is None for value in (*left_values, *right_values)):
        return False
    no_worse = all(a <= b for a, b in zip(left_values, right_values))
    strictly_better = any(a < b for a, b in zip(left_values, right_values))
    return no_worse and strictly_better
