from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.parent_selection import select_refinement_parent


def _candidate(
    output_dir: Path,
    index: int,
    *,
    latency: float,
    latency_delta: float,
    dsp_delta: float,
) -> dict:
    return {
        "candidate_index": index,
        "candidate_file": str(output_dir / f"candidate_{index:03d}.cpp"),
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": True,
        "fully_verified": True,
        "verdict": "keep_pareto_candidate",
        "deltas_percent": {
            "latency_ns": latency_delta,
            "resources_lut_used": 0.0,
            "resources_ff_used": 0.0,
            "resources_dsp_used": dsp_delta,
            "resources_bram_used": 0.0,
        },
        "metrics": {
            "latency_ns": latency,
            "throughput_period_ns": latency + 1.0,
            "resources_lut_used": 100,
            "resources_ff_used": 100,
            "resources_dsp_used": 10,
            "resources_bram_used": 0,
        },
    }


def test_static_strategy_mismatch_terminates_source_recovery(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    best = _candidate(
        output_dir,
        3,
        latency=100.0,
        latency_delta=-1.0,
        dsp_delta=0.0,
    )
    recovery_root = _candidate(
        output_dir,
        7,
        latency=190.0,
        latency_delta=90.0,
        dsp_delta=-50.0,
    )
    failed = {
        "candidate_index": 8,
        "candidate_file": str(output_dir / "candidate_008.cpp"),
        "static_validation": False,
        "csim": None,
        "synthesis": None,
        "cosim": None,
        "fully_verified": False,
        "verdict": "reject_static",
        "metrics": {},
    }
    (output_dir / "candidate_008_strategy.json").write_text(
        json.dumps(
            {
                "name": "recover_latency_tradeoff",
                "parameters": {"factor": 2},
                "source_candidate_index": 7,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "candidate_008_static_validation.json").write_text(
        json.dumps(
            {
                "passed": False,
                "strategy_compliance": {
                    "required": True,
                    "passed": False,
                },
            }
        ),
        encoding="utf-8",
    )

    selected = select_refinement_parent([best, recovery_root, failed])

    assert selected is not None
    assert selected[0]["candidate_index"] == 3
    assert selected[1] == "pareto_candidate"
