from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.archive import preserve_candidate_state
from agent.config import TaskManifest
from agent.optimise.config_source import InMemoryConfig, ppa_config_from_task
from agent.optimise.evaluate import classify_candidate, record_dominates
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


def test_derived_metrics_use_estimated_clock_period() -> None:
    metrics = derive_hardware_metrics(
        _metrics(clock=1.651, latency=18, interval=16),
        minimum_frequency_mhz=100,
    )

    assert metrics["latency_ns"] == pytest.approx(29.718)
    assert metrics["throughput_period_ns"] == pytest.approx(26.416)
    assert metrics["frequency_mhz"] == pytest.approx(605.693519)
    assert metrics["maximum_clock_period_ns"] == pytest.approx(10.0)
    assert metrics["meets_minimum_frequency"] is True


def test_fewer_cycles_are_rejected_when_actual_time_is_worse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    baseline = derive_hardware_metrics(
        _metrics(clock=2.0, latency=10, interval=10),
        minimum_frequency_mhz=100,
    )
    output = _write_candidate(
        tmp_path,
        index=1,
        metrics=_metrics(clock=6.0, latency=5, interval=5),
    )

    record = classify_candidate(
        output,
        1,
        baseline,
        {},
        minimum_frequency_mhz=100,
    )

    assert record["metrics"]["latency_best_cycles"] == 5
    assert record["metrics"]["latency_ns"] == pytest.approx(30.0)
    assert record["performance_comparison"]["latency_metric"] == "latency_ns"
    assert record["verdict"] == "reject_no_performance_gain"


def test_candidate_below_minimum_frequency_is_rejected_before_ppa_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    baseline = derive_hardware_metrics(
        _metrics(clock=2.0, latency=10, interval=10),
        minimum_frequency_mhz=100,
    )
    output = _write_candidate(
        tmp_path,
        index=1,
        metrics=_metrics(clock=12.0, latency=1, interval=1, lut=10, ff=5),
    )

    record = classify_candidate(
        output,
        1,
        baseline,
        {},
        minimum_frequency_mhz=100,
    )

    assert record["metrics"]["frequency_mhz"] == pytest.approx(83.333333)
    assert record["meets_frequency_requirement"] is False
    assert record["verdict"] == "reject_frequency_threshold"
    assert "100 MHz" in record["reason"]


def test_compliant_real_time_improvement_can_dominate_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.evaluate.REPO_ROOT", tmp_path)
    baseline = derive_hardware_metrics(
        _metrics(clock=2.0, latency=10, interval=10),
        minimum_frequency_mhz=100,
    )
    output = _write_candidate(
        tmp_path,
        index=1,
        metrics=_metrics(clock=2.0, latency=8, interval=8),
    )

    record = classify_candidate(
        output,
        1,
        baseline,
        {},
        minimum_frequency_mhz=100,
    )

    assert record["metrics"]["latency_ns"] == pytest.approx(16.0)
    assert record["meets_frequency_requirement"] is True
    assert record["verdict"] == "accept_dominates_baseline"


def test_pareto_dominance_uses_actual_time_instead_of_cycle_count() -> None:
    fewer_cycles_but_slower = {
        "metrics": derive_hardware_metrics(
            _metrics(clock=6.0, latency=5, interval=5),
            minimum_frequency_mhz=100,
        )
    }
    more_cycles_but_faster = {
        "metrics": derive_hardware_metrics(
            _metrics(clock=2.0, latency=10, interval=10),
            minimum_frequency_mhz=100,
        )
    }

    assert not record_dominates(fewer_cycles_but_slower, more_cycles_but_faster)
    assert record_dominates(more_cycles_but_faster, fewer_cycles_but_slower)


def test_task_frequency_requirement_is_forwarded_to_ppa_config() -> None:
    task = TaskManifest(
        path=Path("task.json"),
        data={
            "task_id": "kernel_001",
            "task_root": "benchmarks/kernel",
            "artifacts": {
                "source": "benchmarks/kernel/src/kernel.cpp",
                "testbench": ["benchmarks/kernel/testbench/kernel_tb.cpp"],
                "build_files": ["benchmarks/kernel/task.cfg"],
            },
            "interface": {
                "top_function": "kernel",
                "protected_contract": ["Preserve output semantics."],
            },
            "target": {
                "clock_period_ns": 8.0,
                "minimum_frequency_mhz": 125.0,
                "part": "xczu3eg-sfvc784-2-e",
            },
            "model": {"name": "model"},
            "budgets": {"max_iterations": 2, "max_synthesis_calls": 3},
            "optimisation": {},
            "output_dir": "experiments/kernel",
        },
    )

    config = ppa_config_from_task(task)

    assert config["target_clock_period_ns"] == 8.0
    assert config["minimum_frequency_mhz"] == 125.0


def test_durable_selection_prefers_frequency_compliant_design(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.archive.REPO_ROOT", tmp_path)
    baseline_source = tmp_path / "baseline.cpp"
    candidate_source = tmp_path / "out/candidate_001.cpp"
    candidate_source.parent.mkdir(parents=True)
    baseline_source.write_text("void kernel() {}\n", encoding="utf-8")
    candidate_source.write_text("void kernel() { int x = 1; }\n", encoding="utf-8")

    config = InMemoryConfig(
        {
            "output_dir": "out",
            "baseline": {"source": "baseline.cpp"},
        },
        "selection-test",
    )
    baseline_metrics = derive_hardware_metrics(
        _metrics(clock=12.0, latency=2, interval=2),
        minimum_frequency_mhz=100,
    )
    candidate_metrics = derive_hardware_metrics(
        _metrics(clock=5.0, latency=5, interval=5),
        minimum_frequency_mhz=100,
    )
    candidate = {
        "candidate_index": 1,
        "candidate_file": "out/candidate_001.cpp",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "metrics": candidate_metrics,
        "meets_frequency_requirement": True,
        "verdict": "keep_pareto_candidate",
    }
    summary = {
        "baseline_metrics": baseline_metrics,
        "frequency_requirement": {
            "minimum_frequency_mhz": 100.0,
            "maximum_clock_period_ns": 10.0,
            "baseline_meets_requirement": False,
        },
        "candidates": [candidate],
        "pareto_archive": [candidate],
    }

    enriched = preserve_candidate_state(config, summary)

    assert enriched["selected_design"]["candidate_index"] == 1
    assert enriched["selected_design_frequency_compliant"] is True
