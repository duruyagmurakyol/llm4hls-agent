from __future__ import annotations

import pytest

from agent.optimise.search_policy import (
    EXPLORATION_STRATEGY_FAMILIES,
    MAX_STRUCTURED_CANDIDATES,
    build_structured_search_schedule,
)


def test_five_candidate_schedule_has_three_explorations_then_exploit_recover() -> None:
    schedule = build_structured_search_schedule(max_candidates=5)

    assert len(schedule) == 5
    assert [attempt["candidate_index"] for attempt in schedule] == [1, 2, 3, 4, 5]
    assert [attempt["phase"] for attempt in schedule] == [
        "explore",
        "explore",
        "explore",
        "exploit",
        "recover",
    ]

    first_three = schedule[:3]
    assert tuple(
        attempt["strategy_family"] for attempt in first_three
    ) == EXPLORATION_STRATEGY_FAMILIES
    assert all(
        attempt["parent_candidate_index"] == 0
        for attempt in first_three
    )
    assert all(
        attempt["parent_selector"] == "verified_baseline"
        for attempt in first_three
    )
    assert all(
        attempt["fallback_to_baseline"] is False
        for attempt in first_three
    )

    exploit = schedule[3]
    assert exploit["strategy_family"] == "focused_exploitation"
    assert exploit["parent_candidate_index"] is None
    assert exploit["parent_selector"] == "best_refinement_eligible_candidate"
    assert exploit["fallback_to_baseline"] is True

    recover = schedule[4]
    assert (
        recover["strategy_family"]
        == "bounded_recovery_or_independent_fallback"
    )
    assert recover["parent_candidate_index"] is None
    assert recover["parent_selector"] == "best_recoverable_candidate_or_baseline"
    assert recover["fallback_to_baseline"] is True


@pytest.mark.parametrize("budget", range(0, MAX_STRUCTURED_CANDIDATES + 1))
def test_schedule_respects_smaller_candidate_budgets(budget: int) -> None:
    schedule = build_structured_search_schedule(max_candidates=budget)

    assert len(schedule) == budget
    assert [attempt["candidate_index"] for attempt in schedule] == list(
        range(1, budget + 1)
    )


def test_additional_budget_does_not_create_unstructured_retries() -> None:
    schedule = build_structured_search_schedule(max_candidates=20)

    assert len(schedule) == MAX_STRUCTURED_CANDIDATES
    assert schedule == build_structured_search_schedule(
        max_candidates=MAX_STRUCTURED_CANDIDATES
    )


@pytest.mark.parametrize("value", [True, False, 5.0, "5", None])
def test_non_integer_candidate_budget_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="max_candidates must be an integer"):
        build_structured_search_schedule(max_candidates=value)  # type: ignore[arg-type]


def test_negative_candidate_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_candidates must be non-negative"):
        build_structured_search_schedule(max_candidates=-1)
