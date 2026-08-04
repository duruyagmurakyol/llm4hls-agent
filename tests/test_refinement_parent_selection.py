from __future__ import annotations

from agent.optimise.parent_selection import select_refinement_parent


def test_older_promising_synthesis_candidate_beats_latest_csim_failure() -> None:
    selected = select_refinement_parent(
        [
            {
                "candidate_index": 1,
                "static_validation": True,
                "csim": True,
                "synthesis": True,
                "fully_verified": False,
                "refinement_eligible": True,
                "verdict": "reject_frequency_threshold",
            },
            {
                "candidate_index": 2,
                "static_validation": True,
                "csim": False,
                "synthesis": None,
                "fully_verified": False,
                "refinement_eligible": False,
                "verdict": "reject_csim",
            },
        ]
    )

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 1
    assert reason == "synthesis_passed_refinement_eligible"


def test_duplicate_candidate_is_never_selected() -> None:
    selected = select_refinement_parent(
        [
            {
                "candidate_index": 1,
                "static_validation": True,
                "csim": True,
                "synthesis": None,
                "fully_verified": False,
                "verdict": "reject_csim",
            },
            {
                "candidate_index": 2,
                "static_validation": True,
                "csim": True,
                "synthesis": True,
                "fully_verified": False,
                "refinement_eligible": True,
                "verdict": "reject_duplicate",
            },
        ]
    )

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 1
    assert reason == "csim_passed_candidate"


def test_fully_verified_candidate_beats_promising_synthesis_candidate() -> None:
    selected = select_refinement_parent(
        [
            {
                "candidate_index": 1,
                "static_validation": True,
                "csim": True,
                "synthesis": True,
                "fully_verified": True,
                "refinement_eligible": False,
                "verdict": "keep_pareto_candidate",
            },
            {
                "candidate_index": 2,
                "static_validation": True,
                "csim": True,
                "synthesis": True,
                "fully_verified": False,
                "refinement_eligible": True,
                "verdict": "reject_frequency_threshold",
            },
        ]
    )

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 1
    assert reason == "fully_verified_candidate"


def test_latest_candidate_wins_within_same_quality_tier() -> None:
    selected = select_refinement_parent(
        [
            {
                "candidate_index": 1,
                "static_validation": True,
                "csim": True,
                "synthesis": None,
                "fully_verified": False,
                "verdict": "reject_synthesis_failed",
            },
            {
                "candidate_index": 3,
                "static_validation": True,
                "csim": True,
                "synthesis": None,
                "fully_verified": False,
                "verdict": "reject_synthesis_failed",
            },
        ]
    )

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 3
    assert reason == "csim_passed_candidate"


def test_returns_none_when_no_candidate_is_viable() -> None:
    assert (
        select_refinement_parent(
            [
                {
                    "candidate_index": 1,
                    "static_validation": False,
                    "csim": None,
                    "synthesis": None,
                    "fully_verified": False,
                    "verdict": "reject_static",
                },
                {
                    "candidate_index": 2,
                    "static_validation": True,
                    "csim": None,
                    "synthesis": None,
                    "fully_verified": False,
                    "verdict": "reject_duplicate",
                },
            ]
        )
        is None
    )
