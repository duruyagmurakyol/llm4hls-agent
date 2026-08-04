"""Minimal evidence-directed strategy choice for PPA refinement."""

from __future__ import annotations

from typing import Any


def select_tradeoff_strategy(synthesis_log: str) -> dict[str, Any] | None:
    """Select one legal follow-up for the observed full-unroll bottleneck."""
    lowered = synthesis_log.lower()
    memory_limited = (
        "lower bound of ii" in lowered
        and ("multiple 'load'" in lowered or "multiple 'store'" in lowered)
    )
    full_unroll = "complete unroll" in lowered or (
        "unrolling loop" in lowered and "completely" in lowered
    )
    if not (memory_limited and full_unroll):
        return None

    return {
        "name": "partial_unroll",
        "parameters": {"factor": 8},
        "reason": (
            "The previous complete-unroll design was limited by memory-port "
            "contention, so reduce parallel accesses while retaining useful "
            "loop-level parallelism."
        ),
        "required_changes": [
            "Keep the loop and apply partial unrolling with factor 8.",
            "Preserve legal loop pipelining and the existing interface contract.",
        ],
        "forbidden_changes": [
            "Do not completely unroll the loop.",
            "Do not completely partition top-level interface arrays.",
        ],
    }
