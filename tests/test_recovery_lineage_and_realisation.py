from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.parent_selection import select_refinement_parent
from agent.optimise.pareto_frontier import annotate_pareto_frontier
from agent.optimise.refinement_strategy import check_strategy_compliance


def _pareto_candidate(
    output_dir: Path,
    candidate_index: int,
    *,
    latency_ns: float,
    dsp: int,
    latency_delta_percent: float,
    dsp_delta_percent: float,
) -> dict:
    return {
        "candidate_index": candidate_index,
        "candidate_file": str(output_dir / f"candidate_{candidate_index:03d}.cpp"),
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": True,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {"passed": True},
        "verdict": "keep_pareto_candidate",
        "deltas_percent": {
            "latency_ns": latency_delta_percent,
            "resources_lut_used": 0.0,
            "resources_ff_used": 0.0,
            "resources_dsp_used": dsp_delta_percent,
            "resources_bram_used": 0.0,
        },
        "metrics": {
            "latency_ns": latency_ns,
            "throughput_period_ns": latency_ns + 7.5,
            "resources_lut_used": 8635,
            "resources_ff_used": 23057,
            "resources_dsp_used": dsp,
            "resources_bram_used": 0,
        },
    }


def _write_strategy(
    output_dir: Path,
    candidate_index: int,
    *,
    source_index: int,
    factor: int,
) -> None:
    (output_dir / f"candidate_{candidate_index:03d}_strategy.json").write_text(
        json.dumps(
            {
                "name": "recover_latency_tradeoff",
                "parameters": {"factor": factor},
                "source_candidate_index": source_index,
                "next_candidate_index": candidate_index,
            }
        ),
        encoding="utf-8",
    )


def test_pipeline_and_unroll_must_belong_to_same_innermost_loop() -> None:
    source = """void kernel(double a[8]) {
    for (int i = 0; i < 8; ++i) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < 8; ++j) {
            #pragma HLS UNROLL factor=2
            a[j] += i;
        }
    }
}
"""
    strategy = {
        "name": "recover_latency_tradeoff",
        "parameters": {"factor": 2},
    }

    result = check_strategy_compliance(source, strategy)

    assert result["passed"] is False
    assert result["observed"]["matching_pipeline_unroll_loop"] is False


def test_pipeline_and_unroll_on_same_loop_pass_compliance() -> None:
    source = """void kernel(double a[8]) {
    for (int i = 0; i < 8; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=2
        a[i] += 1;
    }
}
"""
    strategy = {
        "name": "recover_latency_tradeoff",
        "parameters": {"factor": 2},
    }

    result = check_strategy_compliance(source, strategy)

    assert result["passed"] is True
    assert result["observed"]["matching_pipeline_unroll_loop"] is True


def test_recovery_descendant_cannot_become_a_new_recovery_root(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    candidate_3 = _pareto_candidate(
        output_dir,
        3,
        latency_ns=7902.528,
        dsp=56,
        latency_delta_percent=-1.419,
        dsp_delta_percent=0.0,
    )
    candidate_7 = _pareto_candidate(
        output_dir,
        7,
        latency_ns=15240.142,
        dsp=28,
        latency_delta_percent=90.114,
        dsp_delta_percent=-50.0,
    )
    candidate_9 = _pareto_candidate(
        output_dir,
        9,
        latency_ns=15240.142,
        dsp=28,
        latency_delta_percent=90.114,
        dsp_delta_percent=-50.0,
    )
    _write_strategy(output_dir, 8, source_index=7, factor=2)
    _write_strategy(output_dir, 9, source_index=7, factor=4)

    selected = select_refinement_parent(
        [candidate_3, candidate_7, candidate_9]
    )

    assert selected is not None
    assert selected[0]["candidate_index"] == 7
    assert selected[1] == "pending_latency_recovery_strategy"


def test_failed_strategy_realisation_stops_the_remaining_ladder(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    candidate_3 = _pareto_candidate(
        output_dir,
        3,
        latency_ns=7902.528,
        dsp=56,
        latency_delta_percent=-1.419,
        dsp_delta_percent=0.0,
    )
    candidate_7 = _pareto_candidate(
        output_dir,
        7,
        latency_ns=15240.142,
        dsp=28,
        latency_delta_percent=90.114,
        dsp_delta_percent=-50.0,
    )
    candidate_8 = {
        "candidate_index": 8,
        "candidate_file": str(output_dir / "candidate_008.cpp"),
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": None,
        "fully_verified": False,
        "refinement_eligible": False,
        "verdict": "reject_strategy_not_realised",
        "metrics": {},
    }
    _write_strategy(output_dir, 8, source_index=7, factor=2)

    selected = select_refinement_parent(
        [candidate_3, candidate_7, candidate_8]
    )

    assert selected is not None
    assert selected[0]["candidate_index"] == 3
    assert selected[1] == "pareto_candidate"


def test_complete_unroll_conversion_removes_candidate_from_pareto(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    log_path = output_dir / "candidate_008_synthesis.log"
    log_path.write_text(
        "WARNING: [HLS 214-275] partially UNROLL is converted into complete unroll "
        "due to PIPELINE pragma above\n",
        encoding="utf-8",
    )
    _write_strategy(output_dir, 8, source_index=7, factor=2)
    (output_dir / "candidate_008_synthesis.json").write_text(
        json.dumps(
            {
                "passed": True,
                "synthesis_run": True,
                "log_file": str(log_path),
            }
        ),
        encoding="utf-8",
    )

    candidate = _pareto_candidate(
        output_dir,
        8,
        latency_ns=15240.142,
        dsp=28,
        latency_delta_percent=90.114,
        dsp_delta_percent=-50.0,
    )
    summary = {
        "candidates": [candidate],
        "pareto_archive": [dict(candidate)],
    }

    result = annotate_pareto_frontier(output_dir, summary)
    record = result["candidates"][0]

    assert record["verdict"] == "reject_strategy_not_realised"
    assert record["fully_verified"] is False
    assert record["pareto"] is False
    assert record["strategy_realisation"]["passed"] is False
    assert result["pareto_archive"] == []
    assert result["pareto_frontier"]["members"] == []
