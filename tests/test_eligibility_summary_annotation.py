from __future__ import annotations

from agent.optimise.eligibility import annotate_experiment_summary


def _verified_record(
    index: int,
    *,
    verdict: str,
    latency_delta: float,
    lut_delta: float,
    ff_delta: float,
    dsp_delta: float,
) -> dict:
    return {
        "candidate_index": index,
        "candidate_file": f"candidate_{index:03d}.cpp",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim_required": False,
        "cosim": None,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "meets_resource_limits": True,
        "resource_limit_compliance": {
            "configured": True,
            "passed": True,
            "limits": {},
            "usage": {},
            "violations": [],
        },
        "deltas_percent": {
            "latency_ns": latency_delta,
            "throughput_period_ns": latency_delta,
            "resources_lut_used": lut_delta,
            "resources_ff_used": ff_delta,
            "resources_dsp_used": dsp_delta,
            "resources_bram_used": 0.0,
        },
        "verdict": verdict,
    }


def test_summary_keeps_poor_pareto_in_archive_but_not_refinement_pool() -> None:
    baseline = {
        "candidate_index": 0,
        "candidate_file": "baseline.cpp",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim_required": False,
        "cosim": None,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "meets_resource_limits": True,
        "resource_limit_compliance": {
            "configured": True,
            "passed": True,
            "limits": {},
            "usage": {},
            "violations": [],
        },
        "deltas_percent": {},
        "verdict": "baseline",
    }
    poor = _verified_record(
        2,
        verdict="keep_pareto_candidate",
        latency_delta=170.15,
        lut_delta=13.87,
        ff_delta=182.81,
        dsp_delta=-46.15,
    )
    useful = _verified_record(
        3,
        verdict="accept_dominates_baseline",
        latency_delta=-20.0,
        lut_delta=-2.0,
        ff_delta=-1.0,
        dsp_delta=0.0,
    )

    annotated = annotate_experiment_summary(
        {
            "baseline_record": baseline,
            "candidates": [poor, useful],
            "pareto_archive": [baseline, poor, useful],
        }
    )

    by_index = {
        item["candidate_index"]: item
        for item in annotated["candidates"]
    }
    assert by_index[2]["archive_eligible"] is True
    assert by_index[2]["refinement_eligible"] is False
    assert by_index[2]["final_selectable"] is True

    assert by_index[3]["archive_eligible"] is True
    assert by_index[3]["refinement_eligible"] is True
    assert by_index[3]["final_selectable"] is True

    assert annotated["baseline_record"]["archive_eligible"] is True
    assert annotated["baseline_record"]["refinement_eligible"] is True
    assert annotated["baseline_record"]["final_selectable"] is True

    pareto_by_index = {
        item["candidate_index"]: item
        for item in annotated["pareto_archive"]
    }
    assert pareto_by_index[2]["archive_eligible"] is True
    assert pareto_by_index[2]["refinement_eligible"] is False
    assert annotated["eligibility_policy"]["schema_version"] == 1
