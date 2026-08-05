from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.parent_selection import select_refinement_parent


def _pareto_candidate(
    output_dir: Path,
    candidate_index: int,
    *,
    latency_ns: float,
    throughput_period_ns: float,
    lut: int,
    ff: int,
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
            "throughput_period_ns": throughput_period_ns,
            "resources_lut_used": lut,
            "resources_ff_used": ff,
            "resources_dsp_used": dsp,
            "resources_bram_used": 0,
        },
    }


def _run_010_candidates(output_dir: Path) -> tuple[dict, dict]:
    candidate_3 = _pareto_candidate(
        output_dir,
        3,
        latency_ns=7902.528,
        throughput_period_ns=7910.112,
        lut=8873,
        ff=19717,
        dsp=56,
        latency_delta_percent=-1.419,
        dsp_delta_percent=0.0,
    )
    candidate_7 = _pareto_candidate(
        output_dir,
        7,
        latency_ns=15240.142,
        throughput_period_ns=15247.785,
        lut=8635,
        ff=23057,
        dsp=28,
        latency_delta_percent=90.114,
        dsp_delta_percent=-50.0,
    )
    return candidate_3, candidate_7


def _write_recovery_strategy(
    output_dir: Path,
    candidate_index: int,
    factor: int,
) -> None:
    (output_dir / f"candidate_{candidate_index:03d}_strategy.json").write_text(
        json.dumps(
            {
                "name": "recover_latency_tradeoff",
                "parameters": {"factor": factor},
                "source_candidate_index": 7,
                "next_candidate_index": candidate_index,
            }
        ),
        encoding="utf-8",
    )


def test_pending_recovery_side_branch_temporarily_beats_normal_pareto_parent(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    candidate_3, candidate_7 = _run_010_candidates(output_dir)

    selected = select_refinement_parent([candidate_3, candidate_7])

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 7
    assert reason == "pending_latency_recovery_strategy"


def test_side_branch_stays_active_until_factors_two_four_and_eight_complete(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    candidate_3, candidate_7 = _run_010_candidates(output_dir)

    _write_recovery_strategy(output_dir, 8, 2)
    _write_recovery_strategy(output_dir, 9, 4)

    selected = select_refinement_parent([candidate_3, candidate_7])
    assert selected is not None
    assert selected[0]["candidate_index"] == 7
    assert selected[1] == "pending_latency_recovery_strategy"

    _write_recovery_strategy(output_dir, 10, 8)

    selected = select_refinement_parent([candidate_3, candidate_7])
    assert selected is not None
    assert selected[0]["candidate_index"] == 3
    assert selected[1] == "pareto_candidate"


def test_non_applicable_tradeoff_uses_normal_pareto_ranking(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    candidate_3, candidate_7 = _run_010_candidates(output_dir)
    candidate_7["deltas_percent"]["resources_dsp_used"] = -20.0

    selected = select_refinement_parent([candidate_3, candidate_7])

    assert selected is not None
    assert selected[0]["candidate_index"] == 3
    assert selected[1] == "pareto_candidate"
