from __future__ import annotations

import json
from pathlib import Path

from agent.archive import _apply_pre_cosim_pareto_gate


BASELINE_METRICS = {
    "latency_ns": 190.128,
    "throughput_period_ns": 178.944,
    "resources_lut_used": 406,
    "resources_ff_used": 231,
    "resources_dsp_used": 0,
    "resources_bram_used": 0,
}


def _summary(candidate_metrics: dict[str, float | int]) -> dict:
    baseline = {
        "candidate_index": 0,
        "candidate_file": "baseline.cpp",
        "fully_verified": True,
        "metrics": dict(BASELINE_METRICS),
    }
    candidate = {
        "candidate_index": 1,
        "candidate_file": "candidate_001.cpp",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim_required": True,
        "cosim": None,
        "cosim_run": False,
        "fully_verified": False,
        "meets_frequency_requirement": True,
        "meets_resource_limits": True,
        "metrics": candidate_metrics,
        "verdict": "awaiting_cosim",
        "reason": "C/RTL co-simulation is required.",
    }
    return {
        "baseline_record": baseline,
        "pareto_archive": [baseline],
        "candidates": [candidate],
    }


def test_dominated_candidate_skips_expensive_cosim(tmp_path: Path) -> None:
    summary = _summary(
        {
            "latency_ns": 536.832,
            "throughput_period_ns": 525.648,
            "resources_lut_used": 2564,
            "resources_ff_used": 1426,
            "resources_dsp_used": 0,
            "resources_bram_used": 0,
        }
    )

    result = _apply_pre_cosim_pareto_gate(tmp_path, summary)
    record = result["candidates"][0]

    assert record["verdict"] == "reject_dominated_pre_cosim"
    assert record["cosim_skipped"] is True
    assert record["cosim_run"] is False
    assert record["fully_verified"] is False
    assert record["dominated_by"] == 0

    decision = json.loads(
        (tmp_path / "candidate_001_cosim_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["decision"] == "skip_cosim"
    assert decision["dominated_by"] == 0
    assert decision["candidate_metrics"]["latency_ns"] == 536.832


def test_real_tradeoff_still_requires_cosim(tmp_path: Path) -> None:
    summary = _summary(
        {
            "latency_ns": 150.0,
            "throughput_period_ns": 140.0,
            "resources_lut_used": 900,
            "resources_ff_used": 500,
            "resources_dsp_used": 0,
            "resources_bram_used": 0,
        }
    )

    result = _apply_pre_cosim_pareto_gate(tmp_path, summary)
    record = result["candidates"][0]

    assert record["verdict"] == "awaiting_cosim"
    assert "cosim_skipped" not in record
    assert not (tmp_path / "candidate_001_cosim_decision.json").exists()


def test_identical_candidate_skips_cosim(tmp_path: Path) -> None:
    summary = _summary(dict(BASELINE_METRICS))

    result = _apply_pre_cosim_pareto_gate(tmp_path, summary)
    record = result["candidates"][0]

    assert record["verdict"] == "reject_no_change_pre_cosim"
    assert record["cosim_skipped"] is True
    assert record["dominated_by"] == 0
