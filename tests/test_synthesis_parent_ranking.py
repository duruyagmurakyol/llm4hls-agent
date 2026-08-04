from __future__ import annotations

from agent.optimise.parent_selection import select_refinement_parent


def _frequency_violation(
    candidate_index: int,
    *,
    latency_ns: float,
    throughput_period_ns: float,
    lut: int,
    ff: int,
    dsp: int,
) -> dict:
    return {
        "candidate_index": candidate_index,
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "fully_verified": False,
        "refinement_eligible": True,
        "meets_frequency_requirement": False,
        "resource_limit_compliance": {"passed": True},
        "verdict": "reject_frequency_threshold",
        "metrics": {
            "latency_ns": latency_ns,
            "throughput_period_ns": throughput_period_ns,
            "resources_lut_used": lut,
            "resources_ff_used": ff,
            "resources_dsp_used": dsp,
            "resources_bram_used": 0,
        },
    }


def test_best_frequency_violation_beats_newer_worse_candidates() -> None:
    selected = select_refinement_parent(
        [
            _frequency_violation(
                2,
                latency_ns=63458.276,
                throughput_period_ns=63496.689,
                lut=2468,
                ff=1985,
                dsp=29,
            ),
            _frequency_violation(
                3,
                latency_ns=160614.08,
                throughput_period_ns=160711.54,
                lut=4077,
                ff=2665,
                dsp=34,
            ),
            _frequency_violation(
                5,
                latency_ns=162744.876,
                throughput_period_ns=162782.865,
                lut=2250,
                ff=1648,
                dsp=17,
            ),
        ]
    )

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 2
    assert reason == "synthesis_passed_refinement_eligible"


def test_csim_only_tier_still_prefers_latest_feedback() -> None:
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
