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


def test_pareto_candidate_beats_promising_synthesis_candidate() -> None:
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
    assert reason == "pareto_candidate"


def test_pareto_candidate_beats_newer_dominated_verified_candidates() -> None:
    selected = select_refinement_parent(
        [
            {
                "candidate_index": 4,
                "static_validation": True,
                "csim": True,
                "synthesis": True,
                "fully_verified": True,
                "verdict": "keep_pareto_candidate",
            },
            {
                "candidate_index": 5,
                "static_validation": True,
                "csim": True,
                "synthesis": True,
                "fully_verified": True,
                "verdict": "reject_dominated",
            },
            {
                "candidate_index": 6,
                "static_validation": True,
                "csim": True,
                "synthesis": True,
                "fully_verified": True,
                "verdict": "reject_dominated",
            },
            {
                "candidate_index": 7,
                "static_validation": True,
                "csim": True,
                "synthesis": True,
                "fully_verified": True,
                "verdict": "reject_dominated",
            },
        ]
    )

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 4
    assert reason == "pareto_candidate"


def test_promising_constraint_violation_beats_generic_verified_candidate() -> None:
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
                "csim": True,
                "synthesis": True,
                "fully_verified": True,
                "refinement_eligible": False,
                "verdict": "reject_no_objective_gain",
            },
        ]
    )

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 1
    assert reason == "synthesis_passed_refinement_eligible"


def test_dominating_candidate_beats_newer_pareto_candidate() -> None:
    selected = select_refinement_parent(
        [
            {
                "candidate_index": 1,
                "static_validation": True,
                "csim": True,
                "synthesis": True,
                "fully_verified": True,
                "verdict": "accept_dominates_baseline",
            },
            {
                "candidate_index": 2,
                "static_validation": True,
                "csim": True,
                "synthesis": True,
                "fully_verified": True,
                "verdict": "keep_pareto_candidate",
            },
        ]
    )

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 1
    assert reason == "dominates_baseline_candidate"


def test_default_final_ranking_selects_candidate_6_over_newer_candidate_7() -> None:
    candidate_6 = {
        "candidate_index": 6,
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": True,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {"passed": True},
        "verdict": "keep_pareto_candidate",
        "metrics": {
            "latency_ns": 7712.928,
            "throughput_period_ns": 7720.512,
            "resources_lut_used": 9996,
            "resources_ff_used": 21911,
            "resources_dsp_used": 56,
            "resources_bram_used": 0,
        },
    }
    candidate_7 = {
        "candidate_index": 7,
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": True,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {"passed": True},
        "verdict": "keep_pareto_candidate",
        "metrics": {
            "latency_ns": 15240.142,
            "throughput_period_ns": 15247.785,
            "resources_lut_used": 9135,
            "resources_ff_used": 23057,
            "resources_dsp_used": 28,
            "resources_bram_used": 0,
        },
    }

    selected = select_refinement_parent([candidate_6, candidate_7])

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 6
    assert reason == "pareto_candidate"


def test_configured_resource_first_ranking_can_select_candidate_7() -> None:
    candidate_6 = {
        "candidate_index": 6,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {"passed": True},
        "verdict": "keep_pareto_candidate",
        "metrics": {
            "latency_ns": 7712.928,
            "throughput_period_ns": 7720.512,
            "resources_lut_used": 9996,
            "resources_ff_used": 21911,
            "resources_dsp_used": 56,
            "resources_bram_used": 0,
        },
    }
    candidate_7 = {
        "candidate_index": 7,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {"passed": True},
        "verdict": "keep_pareto_candidate",
        "metrics": {
            "latency_ns": 15240.142,
            "throughput_period_ns": 15247.785,
            "resources_lut_used": 9135,
            "resources_ff_used": 23057,
            "resources_dsp_used": 28,
            "resources_bram_used": 0,
        },
    }

    selected = select_refinement_parent(
        [candidate_6, candidate_7],
        {"ranking": ["resource_cost", "latency_ns"]},
    )

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 7
    assert reason == "pareto_candidate"


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


def test_static_rejection_is_retained_as_last_resort_feedback_parent() -> None:
    selected = select_refinement_parent(
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

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 1
    assert reason == "latest_non_duplicate_fallback"


def test_returns_none_when_every_candidate_is_a_duplicate() -> None:
    assert (
        select_refinement_parent(
            [
                {
                    "candidate_index": 1,
                    "static_validation": True,
                    "csim": None,
                    "synthesis": None,
                    "fully_verified": False,
                    "verdict": "reject_duplicate",
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
