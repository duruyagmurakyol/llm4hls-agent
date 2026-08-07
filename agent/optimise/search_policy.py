"""Structured, budget-bounded optimisation search schedule.

This module defines *what kind* of candidate each search slot should attempt.
It deliberately does not choose concrete source files, generate prompts, call a
model, or run validation tools. The five-slot policy is:

1-3. Explore three diagnosis-selected strategy families from the verified baseline.
4. Exploit the best practically refinement-eligible candidate.
5. Perform one bounded recovery, or fall back to an independent baseline attempt.

For backwards compatibility, callers that do not provide diagnosis-selected
families receive the original three exploration families exactly as before.

The schedule is capped at five candidates. Additional task budget remains
available to the surrounding Track-A controller, but this policy will not invent
unstructured retries merely because more candidate slots exist.

Importing this module also installs the idempotent structured-search transition
and novelty-ledger guards. Those guards are inert for legacy configurations.
"""

from __future__ import annotations

from typing import Any, Iterable

MAX_STRUCTURED_CANDIDATES = 5

# Preserve the original exploration tuple as the compatibility/default policy.
DEFAULT_EXPLORATION_STRATEGY_FAMILIES = (
    "critical_path_restructuring",
    "bounded_unroll",
    "memory_parallelism",
)
EXPLORATION_STRATEGY_FAMILIES = DEFAULT_EXPLORATION_STRATEGY_FAMILIES

# Layer one contains only generic HLS architectural mechanisms.  Nothing here
# names or special-cases an individual benchmark.
LAYER_ONE_STRATEGY_FAMILIES = (
    "critical_path_restructuring",
    "bounded_unroll",
    "memory_parallelism",
    "buffered_parallelism",
    "sliding_window_reuse",
    "dataflow_pipeline",
)


def _normalise_exploration_families(
    families: Iterable[str] | None,
) -> tuple[str, str, str]:
    """Return exactly three valid, distinct exploration families.

    ``None`` intentionally preserves the historical fixed schedule.  Diagnosis-
    aware callers may supply another ordered triple from the generic layer-one
    pool.  Invalid or duplicate selections fail early rather than silently
    changing search behaviour.
    """

    if families is None:
        return DEFAULT_EXPLORATION_STRATEGY_FAMILIES

    selected = tuple(families)
    if len(selected) != 3:
        raise ValueError("structured exploration requires exactly three strategy families")
    if len(set(selected)) != 3:
        raise ValueError("structured exploration strategy families must be distinct")

    unsupported = [
        family for family in selected if family not in LAYER_ONE_STRATEGY_FAMILIES
    ]
    if unsupported:
        raise ValueError(
            "unsupported layer-one exploration strategies: " + ", ".join(unsupported)
        )
    return selected  # type: ignore[return-value]


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
    exploration_strategy_families: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the deterministic search-slot schedule for a candidate budget.

    ``parent_candidate_index`` is concrete for the three exploration slots.
    Later slots use ``None`` because their parent must be resolved from measured
    results at runtime using ``parent_selector``.

    The optional ``exploration_strategy_families`` hook is deliberately narrow:
    it may change only which three generic layer-one mechanisms occupy C1-C3.
    Parent selection, C4 exploitation and C5 recovery remain unchanged.
    """

    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise TypeError("max_candidates must be an integer")
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")

    exploration_families = _normalise_exploration_families(
        exploration_strategy_families
    )
    budget = min(max_candidates, MAX_STRUCTURED_CANDIDATES)
    schedule: list[dict[str, Any]] = []

    for strategy_family in exploration_families:
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


# Install once after the pure schedule API is defined. The transition guard
# preserves the C1->C2->C3 baseline-rooted schedule after an early rejection;
# the ledger guards enforce branch and source novelty. Both bypass legacy runs.
from agent.optimise.structured_transition_runtime import (  # noqa: E402
    install_structured_transition_runtime,
)
from agent.optimise.search_ledger_runtime import (  # noqa: E402
    install_structured_search_ledger_runtime,
)

install_structured_transition_runtime()
install_structured_search_ledger_runtime()
