"""Minimal evidence-directed strategy choice for PPA refinement."""

from __future__ import annotations

import re
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


def check_strategy_compliance(source: str, strategy: dict[str, Any]) -> dict[str, Any]:
    """Check the one strategy currently supported by evidence-directed refinement."""
    name = strategy.get("name")
    if name != "partial_unroll":
        return {
            "required": True,
            "passed": False,
            "strategy": name,
            "reason": "unsupported_strategy",
        }

    factor = int(strategy.get("parameters", {}).get("factor", 0))
    pragmas = re.findall(r"#\s*pragma\s+HLS\s+UNROLL\b([^\n]*)", source, re.I)
    observed_factors = [
        int(match.group(1))
        for arguments in pragmas
        if (match := re.search(r"\bfactor\s*=\s*(\d+)", arguments, re.I))
    ]
    complete_unroll = any(
        not re.search(r"\bfactor\s*=\s*\d+", arguments, re.I)
        for arguments in pragmas
    )
    return {
        "required": True,
        "passed": factor > 0 and factor in observed_factors and not complete_unroll,
        "strategy": name,
        "expected": {"factor": factor},
        "observed": {
            "unroll_factors": observed_factors,
            "complete_unroll": complete_unroll,
        },
    }
