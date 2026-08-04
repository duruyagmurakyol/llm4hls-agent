from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.optimise.config_source import InMemoryConfig
from agent.optimise.evaluate import (
    PARETO_ELIGIBLE_VERDICTS,
    classify_candidate,
    evaluate_experiment,
)
from agent.optimise.metrics import derive_hardware_metrics


def _metrics(
    *,
    clock: float,
    latency: int,
    interval: int,
    lut: int = 100,
    ff: int = 20,
    dsp: int = 0,
    bram: int = 0,
) -> dict[str, object]:
    return {
        "clock_period_ns": clock,
        "latency_best_cycles": latency,
        "latency_average_cycles": latency,
        "latency_worst_cycles": latency,
        "interval_min_cycles": interval,
        "interval_max_cycles": interval,
        "resources_lut_used": lut,
        "resources_ff_used": ff,
        "resources_dsp_used": dsp,
        "resources_bram_used": bram,
    }


def _write_candidate(
    root: Path,
    *,
    index: int,
    metrics: dict[str, object],
) -> Path:
    output = root / "out"
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"candidate_{index:03d}"
    (output / f"{prefix}.cpp").write_text(
        f"void kernel() {{ /* candidate {index} */ }}\n",
        encoding="utf-8",
    )
    (output / f"{prefix}_static_validation.json").write_text(
        json.dumps({"passed": True, "checks": {"source": True}}),
        encoding="utf-8",
    )
    (output / f"{prefix}_csim_validation.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    (output / f"{prefix}_synthesis.json").write_text(
        json.dumps(
            {
                "passed": True,
                "synthesis_run": True,
                "timed_out": False,
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    return output


def _baseline() -> dict[str, object]:
    return derive_hardware_metrics(
        _metrics(clock=2.0, latency=10, interval=10),
        minimum_frequency_mhz=100,
    )


def test_frequency_rejection_preserves_baseline_relative_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    output = _write_candidate(
        tmp_path,
        index=1,
        metrics=_metrics(
            clock=12.0,
            latency=1,
            interval=1,
            lut=10,
            ff=5,
            dsp=1,
        ),
    )

    record = classify_candidate(
        output,
        1,
        _baseline(),
        {},
        minimum_frequency_mhz=100,
    )

    assert record["verdict"] == "reject_frequency_threshold"
    assert record["fully_verified"] is False
    assert record["performance_comparison"]["latency_delta_percent"] == pytest.approx(-40.0)
    assert record["performance_comparison"]["throughput_delta_percent"] == pytest.approx(-40.0)
    assert record["deltas_percent"]["resources_lut_used"] == pytest.approx(-90.0)
    assert record["deltas_percent"]["resources_ff_used"] == pytest.approx(-75.0)
    assert record["deltas_percent"]["resources_dsp_used"] is None
    assert record["deltas_percent"]["resources_bram_used"] == pytest.approx(0.0)
    assert record["usefulness_classification"] == "promising_constraint_violation"
    assert record["refinement_eligible"] is True
    assert record["verdict"] not in PARETO_ELIGIBLE_VERDICTS


def test_resource_limit_rejection_preserves_baseline_relative_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    output = _write_candidate(
        tmp_path,
        index=1,
        metrics=_metrics(clock=2.0, latency=8, interval=8, lut=150, ff=10),
    )

    record = classify_candidate(
        output,
        1,
        _baseline(),
        {},
        minimum_frequency_mhz=100,
        resource_limits={"lut": 120},
    )

    assert record["verdict"] == "reject_resource_limits"
    assert record["meets_resource_limits"] is False
    assert record["performance_comparison"]["latency_delta_percent"] == pytest.approx(-20.0)
    assert record["performance_comparison"]["throughput_delta_percent"] == pytest.approx(-20.0)
    assert record["deltas_percent"]["resources_lut_used"] == pytest.approx(50.0)
    assert record["deltas_percent"]["resources_ff_used"] == pytest.approx(-50.0)
    assert record["usefulness_classification"] == "promising_constraint_violation"
    assert record["refinement_eligible"] is True
    assert record["verdict"] not in PARETO_ELIGIBLE_VERDICTS


def test_small_resource_reduction_is_not_refinement_eligible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    output = _write_candidate(
        tmp_path,
        index=1,
        metrics=_metrics(clock=12.0, latency=1, interval=1, lut=90, ff=19),
    )

    record = classify_candidate(
        output,
        1,
        _baseline(),
        {},
        minimum_frequency_mhz=100,
    )

    assert record["verdict"] == "reject_frequency_threshold"
    assert record["usefulness_classification"] == "constraint_violation"
    assert record["refinement_eligible"] is False


def test_frequency_rejected_candidate_remains_outside_pareto_archive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    baseline_source = tmp_path / "baseline.cpp"
    baseline_source.write_text("void kernel() {}\n", encoding="utf-8")
    _write_candidate(
        tmp_path,
        index=1,
        metrics=_metrics(clock=12.0, latency=1, interval=1, lut=10, ff=5),
    )
    config = InMemoryConfig(
        {
            "experiment_name": "comparison-evidence",
            "benchmark": "kernel",
            "top_function": "kernel",
            "minimum_frequency_mhz": 100.0,
            "resource_limits": {},
            "selection": {},
            "baseline": {
                "source": "baseline.cpp",
                "metrics": _metrics(clock=2.0, latency=10, interval=10),
                "verification": {
                    "csim_passed": True,
                    "synthesis_passed": True,
                    "cosim_passed": True,
                },
            },
            "budget": {
                "max_candidates": 1,
                "max_synthesis_calls": 1,
                "max_cosim_calls": 1,
            },
            "output_dir": "out",
        },
        "comparison-evidence",
    )

    summary = evaluate_experiment(config)

    rejected = summary["candidates"][0]
    assert rejected["verdict"] == "reject_frequency_threshold"
    assert rejected["performance_comparison"]["latency_delta_percent"] == pytest.approx(-40.0)
    assert rejected["usefulness_classification"] == "promising_constraint_violation"
    assert rejected["refinement_eligible"] is True
    assert [item["candidate_index"] for item in summary["pareto_archive"]] == [0]
