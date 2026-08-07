from __future__ import annotations

from agent.optimise.parent_selection import select_refinement_parent
from agent.optimise.synthesis_equivalence import (
    apply_synthesis_equivalence,
    synthesis_equivalence_evidence,
)


def _metrics(*, latency_cycles: int = 7651) -> dict[str, float | int]:
    return {
        "clock_period_ns": 7.445,
        "frequency_mhz": 134.31833445265278,
        "latency_best_cycles": latency_cycles,
        "latency_average_cycles": latency_cycles,
        "latency_worst_cycles": latency_cycles,
        "latency_ns": latency_cycles * 7.445,
        "interval_min_cycles": 7664,
        "interval_max_cycles": 7664,
        "throughput_period_ns": 57058.48,
        "resources_lut_used": 5419,
        "resources_ff_used": 8384,
        "resources_dsp_used": 62,
        "resources_bram_used": 0,
    }


def _record(index: int, metrics: dict[str, float | int]) -> dict[str, object]:
    return {
        "candidate_index": index,
        "candidate_file": f"out/candidate_{index:03d}.cpp",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": True,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {"passed": True, "violations": []},
        "meets_resource_limits": True,
        "metrics": metrics,
        "refinement_eligible": False,
        "verdict": "keep_pareto_candidate",
    }


def test_report_rounding_difference_is_synthesis_equivalent() -> None:
    earlier = _metrics()
    later = {**_metrics(), "throughput_period_ns": 57058.464}

    evidence = synthesis_equivalence_evidence(later, earlier)

    assert evidence is not None
    assert evidence["timing_basis"] == "cycles_and_clock"
    assert evidence["exact_cycle_and_resource_counts"] is True


def test_one_cycle_change_remains_a_distinct_hardware_result() -> None:
    assert synthesis_equivalence_evidence(
        _metrics(latency_cycles=7650),
        _metrics(latency_cycles=7651),
    ) is None


def test_later_equivalent_candidate_is_removed_from_pareto() -> None:
    candidate_2 = _record(2, _metrics())
    candidate_9 = _record(
        9,
        {**_metrics(), "throughput_period_ns": 57058.464},
    )
    baseline = {
        **_record(0, {
            **_metrics(latency_cycles=16021),
            "interval_min_cycles": 16024,
            "interval_max_cycles": 16024,
            "throughput_period_ns": 119302.0,
            "resources_lut_used": 1781,
            "resources_ff_used": 1635,
            "resources_dsp_used": 26,
        }),
        "candidate_file": "baseline.cpp",
        "verdict": "baseline",
    }
    summary = {
        "schema_version": 7,
        "baseline_record": baseline,
        "candidates": [candidate_2, candidate_9],
        "pareto_archive": [baseline, candidate_2, candidate_9],
    }

    updated = apply_synthesis_equivalence(summary)
    records = {
        item["candidate_index"]: item
        for item in updated["candidates"]
    }

    assert records[2]["verdict"] == "keep_pareto_candidate"
    assert records[9]["verdict"] == "reject_synthesis_equivalent"
    assert records[9]["synthesis_equivalent_to"] == 2
    assert records[9]["pareto"] is False
    assert [
        item["candidate_index"] for item in updated["pareto_archive"]
    ] == [0, 2]
    assert updated["schema_version"] == 8


def test_synthesis_equivalent_retired_results_restart_from_baseline() -> None:
    useful = _record(2, _metrics())
    equivalent = {
        **_record(9, _metrics()),
        "verdict": "reject_synthesis_equivalent",
        "synthesis_equivalent_to": 2,
    }

    selected = select_refinement_parent([useful, equivalent])

    assert selected is not None
    parent, reason = selected
    assert parent["candidate_index"] == 0
    assert reason == "restart_from_verified_baseline"
