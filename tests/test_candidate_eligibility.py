from __future__ import annotations

from agent.optimise.eligibility import candidate_eligibility


def _record(
    *,
    verdict: str,
    latency_delta: float,
    throughput_delta: float | None = None,
    lut_delta: float = 0.0,
    ff_delta: float = 0.0,
    dsp_delta: float = 0.0,
    fully_verified: bool = True,
    frequency_ok: bool = True,
    resources_ok: bool = True,
    requires_cosim: bool = False,
    cosim: bool | None = None,
) -> dict:
    return {
        "candidate_index": 1,
        "verdict": verdict,
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim_required": requires_cosim,
        "cosim": cosim,
        "fully_verified": fully_verified,
        "meets_frequency_requirement": frequency_ok,
        "meets_resource_limits": resources_ok,
        "resource_limit_compliance": {
            "configured": True,
            "passed": resources_ok,
            "limits": {},
            "usage": {},
            "violations": [],
        },
        "deltas_percent": {
            "latency_ns": latency_delta,
            "throughput_period_ns": (
                latency_delta if throughput_delta is None else throughput_delta
            ),
            "resources_lut_used": lut_delta,
            "resources_ff_used": ff_delta,
            "resources_dsp_used": dsp_delta,
            "resources_bram_used": 0.0,
        },
    }


def test_dsp_only_extreme_tradeoff_is_archived_but_not_refined() -> None:
    flags = candidate_eligibility(
        _record(
            verdict="keep_pareto_candidate",
            latency_delta=170.15,
            lut_delta=13.87,
            ff_delta=182.81,
            dsp_delta=-46.15,
        )
    )

    assert flags == {
        "archive_eligible": True,
        "refinement_eligible": False,
        "final_selectable": True,
    }


def test_material_latency_improvement_is_archived_refined_and_selectable() -> None:
    flags = candidate_eligibility(
        _record(
            verdict="accept_dominates_baseline",
            latency_delta=-29.09,
            throughput_delta=-28.5,
            lut_delta=-4.55,
            ff_delta=-2.14,
        )
    )

    assert flags == {
        "archive_eligible": True,
        "refinement_eligible": True,
        "final_selectable": True,
    }


def test_small_latency_regression_with_material_lut_saving_can_be_refined() -> None:
    flags = candidate_eligibility(
        _record(
            verdict="keep_pareto_candidate",
            latency_delta=5.0,
            throughput_delta=5.0,
            lut_delta=-20.0,
            ff_delta=-5.0,
        )
    )

    assert flags == {
        "archive_eligible": True,
        "refinement_eligible": True,
        "final_selectable": True,
    }


def test_no_gain_candidate_has_no_eligibility() -> None:
    flags = candidate_eligibility(
        _record(
            verdict="reject_no_objective_gain",
            latency_delta=5.38,
            lut_delta=30.43,
            ff_delta=35.17,
            dsp_delta=7.69,
        )
    )

    assert flags == {
        "archive_eligible": False,
        "refinement_eligible": False,
        "final_selectable": False,
    }


def test_fast_resource_violation_is_recoverable_but_not_archivable() -> None:
    flags = candidate_eligibility(
        _record(
            verdict="reject_resource_limits",
            latency_delta=-38.85,
            throughput_delta=-38.8,
            lut_delta=153.3,
            ff_delta=432.2,
            dsp_delta=7.69,
            resources_ok=False,
        )
    )

    assert flags == {
        "archive_eligible": False,
        "refinement_eligible": True,
        "final_selectable": False,
    }


def test_final_selection_requires_requested_cosim() -> None:
    awaiting = candidate_eligibility(
        _record(
            verdict="keep_pareto_candidate",
            latency_delta=-10.0,
            requires_cosim=True,
            cosim=None,
            fully_verified=False,
        )
    )
    passed = candidate_eligibility(
        _record(
            verdict="keep_pareto_candidate",
            latency_delta=-10.0,
            requires_cosim=True,
            cosim=True,
            fully_verified=True,
        )
    )

    assert awaiting == {
        "archive_eligible": False,
        "refinement_eligible": False,
        "final_selectable": False,
    }
    assert passed == {
        "archive_eligible": True,
        "refinement_eligible": True,
        "final_selectable": True,
    }


def test_verified_baseline_is_all_three_eligibilities() -> None:
    flags = candidate_eligibility(
        _record(
            verdict="baseline",
            latency_delta=0.0,
            lut_delta=0.0,
            ff_delta=0.0,
        )
    )

    assert flags == {
        "archive_eligible": True,
        "refinement_eligible": True,
        "final_selectable": True,
    }
