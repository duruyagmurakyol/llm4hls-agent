#!/usr/bin/env python3

from __future__ import annotations

from typing import Any


INVALID_SCORE = -1.0e30


def score_candidate(result: dict[str, Any]) -> float:
    """
    Rank synthesized candidates by timing, latency, and resource usage.

    Higher scores are better. Timing failures remain rankable so the search can
    distinguish candidates that are closer to meeting the target clock.
    """

    if not result.get("csim_pass", False):
        return INVALID_SCORE

    if not result.get("synth_pass", False):
        return INVALID_SCORE

    metrics = result.get("metrics", {})

    try:
        clock = float(metrics["estimated_clock_period_ns"])
        target = float(metrics["target_clock_period_ns"])
        latency = float(metrics["latency_worst_cycles"])

        lut = float(metrics["resources_lut_used"])
        ff = float(metrics["resources_ff_used"])
        dsp = float(metrics["resources_dsp_used"])
        bram = float(metrics["resources_bram_used"])
    except (KeyError, TypeError, ValueError):
        return INVALID_SCORE

    timing_violation = max(0.0, clock - target)

    # Timing is the dominant constraint.
    # Lower latency and resource usage are better.
    return -(
        timing_violation * 1_000_000.0
        + latency
        + lut * 0.1
        + ff * 0.01
        + dsp * 10.0
        + bram * 10.0
    )
