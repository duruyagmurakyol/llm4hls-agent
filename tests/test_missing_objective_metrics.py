from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.evaluate import (
    PARETO_ELIGIBLE_VERDICTS,
    classify_candidate,
)
from agent.optimise.metrics import derive_hardware_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_fully_verified_resource_saving_candidate_with_missing_timing_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    output = tmp_path / "out"
    output.mkdir()

    (output / "candidate_001.cpp").write_text(
        "void kernel(int a[8]) { for (int i = 0; i < 8; ++i) a[i] += 1; }\n",
        encoding="utf-8",
    )
    _write_json(output / "candidate_001_static_validation.json", {"passed": True})
    _write_json(output / "candidate_001_csim_validation.json", {"passed": True})
    _write_json(
        output / "candidate_001_synthesis.json",
        {
            "passed": True,
            "synthesis_run": True,
            "metrics": {
                "clock_period_ns": 7.5,
                "resources_lut_used": 2910,
                "resources_ff_used": 2106,
                "resources_dsp_used": 28,
                "resources_bram_used": 0,
            },
        },
    )
    _write_json(
        output / "candidate_001_cosim.json",
        {"passed": True, "cosim_run": True},
    )

    baseline = derive_hardware_metrics(
        {
            "clock_period_ns": 7.584,
            "latency_best_cycles": 1057,
            "interval_min_cycles": 1058,
            "resources_lut_used": 8741,
            "resources_ff_used": 21727,
            "resources_dsp_used": 56,
            "resources_bram_used": 0,
        },
        minimum_frequency_mhz=100.0,
    )

    record = classify_candidate(
        output,
        1,
        baseline,
        {},
        minimum_frequency_mhz=100.0,
    )

    assert record["fully_verified"] is True
    assert record["verdict"] == "reject_objective_metrics_unavailable"
    assert record["verdict"] not in PARETO_ELIGIBLE_VERDICTS
    assert record["missing_objectives"] == [
        "latency_best_cycles",
        "interval_min_cycles",
    ]
    assert "latency_best_cycles" in record["reason"]
    assert "interval_min_cycles" in record["reason"]
