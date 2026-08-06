"""Structured, budget-bounded optimisation search schedule.

This module defines *what kind* of candidate each search slot should attempt.
It deliberately does not choose concrete source files, generate prompts, call a
model, or run validation tools. The five-slot policy is:

1. Explore critical-path / accumulator restructuring from the verified baseline.
2. Explore bounded unrolling from the verified baseline.
3. Explore memory parallelism from the verified baseline.
4. Exploit the best practically refinement-eligible candidate.
5. Perform one bounded recovery, or fall back to an independent baseline attempt.

The schedule is capped at five candidates. Additional task budget remains
available to the surrounding Track-A controller, but this policy will not invent
unstructured retries merely because more candidate slots exist.

Importing this module also installs the idempotent structured-search novelty
ledger guards. Those guards are inert for legacy configurations.
"""

from __future__ import annotations

from typing import Any

MAX_STRUCTURED_CANDIDATES = 5

EXPLORATION_STRATEGY_FAMILIES = (
    "critical_path_restructuring",
    "bounded_unroll",
    "memory_parallelism",
)


def _attempt(
    candidate_index: int,
    *,
    phase: str,
    strategy_family: str,
    parent_candidate_index: int | None,
    parent_selector: str,
    fallback_to_baseline: bool,
) -> dict[str, Any]:
    return {
        "candidate_index": candidate_index,
        "phase": phase,
        "strategy_family": strategy_family,
        "parent_candidate_index": parent_candidate_index,
        "parent_selector": parent_selector,
        "fallback_to_baseline": fallback_to_baseline,
    }


def build_structured_search_schedule(
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Return the deterministic search-slot schedule for a candidate budget.

    ``parent_candidate_index`` is concrete for the three exploration slots.
    Later slots use ``None`` because their parent must be resolved from measured
    results at runtime using ``parent_selector``.
    """

    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise TypeError("max_candidates must be an integer")
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")

    budget = min(max_candidates, MAX_STRUCTURED_CANDIDATES)
    schedule: list[dict[str, Any]] = []

    for strategy_family in EXPLORATION_STRATEGY_FAMILIES:
        if len(schedule) >= budget:
            return schedule
        schedule.append(
            _attempt(
                len(schedule) + 1,
                phase="explore",
                strategy_family=strategy_family,
                parent_candidate_index=0,
                parent_selector="verified_baseline",
                fallback_to_baseline=False,
            )
        )

    if len(schedule) < budget:
        schedule.append(
            _attempt(
                len(schedule) + 1,
                phase="exploit",
                strategy_family="focused_exploitation",
                parent_candidate_index=None,
                parent_selector="best_refinement_eligible_candidate",
                fallback_to_baseline=True,
            )
        )

    if len(schedule) < budget:
        schedule.append(
            _attempt(
                len(schedule) + 1,
                phase="recover",
                strategy_family="bounded_recovery_or_independent_fallback",
                parent_candidate_index=None,
                parent_selector="best_recoverable_candidate_or_baseline",
                fallback_to_baseline=True,
            )
        )

    return schedule


# Install once after the pure schedule API is defined. The installer patches the
# preserved runner references, not this policy module, and bypasses every legacy
# configuration that does not opt into ``structured_v1`` or Track-A.
from agent.optimise.search_ledger_runtime import (  # noqa: E402
    install_structured_search_ledger_runtime,
)

install_structured_search_ledger_runtime()
