from __future__ import annotations

import json
from pathlib import Path

from agent import archive


def _metrics() -> dict[str, float | int]:
    return {
        "clock_period_ns": 5.0,
        "frequency_mhz": 200.0,
        "minimum_frequency_mhz": 100.0,
        "maximum_clock_period_ns": 10.0,
        "latency_best_cycles": 10,
        "latency_average_cycles": 10,
        "latency_worst_cycles": 10,
        "latency_ns": 50.0,
        "latency_best_ns": 50.0,
        "latency_average_ns": 50.0,
        "latency_worst_ns": 50.0,
        "interval_min_cycles": 10,
        "interval_max_cycles": 10,
        "throughput_period_ns": 50.0,
        "throughput_period_min_ns": 50.0,
        "throughput_period_max_ns": 50.0,
        "resources_lut_used": 100,
        "resources_ff_used": 50,
        "resources_dsp_used": 1,
        "resources_bram_used": 0,
    }


def test_search_valid_baseline_does_not_invent_cosim_evidence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(archive, "REPO_ROOT", tmp_path)

    baseline = tmp_path / "baseline.cpp"
    baseline.write_text("int kernel(int x) { return x; }\n", encoding="utf-8")

    config = tmp_path / "task.json"
    config.write_text(
        json.dumps(
            {
                "output_dir": "out",
                "baseline": {"source": "baseline.cpp"},
                "requires_cosim": False,
                "selection": {"mode": "research_pareto"},
            }
        ),
        encoding="utf-8",
    )

    enriched = archive.preserve_candidate_state(
        config,
        {
            "baseline_metrics": _metrics(),
            "frequency_requirement": {
                "minimum_frequency_mhz": 100.0,
                "baseline_meets_requirement": True,
            },
            "resource_limits": {},
            "candidates": [],
            "pareto_archive": [],
        },
    )

    state = enriched["candidate_state"]
    selected = state["selected_design"]
    original = state["original_baseline"]

    assert selected["candidate_index"] == 0
    assert selected["fully_verified"] is True
    assert selected["validation"]["csim"] is True
    assert selected["validation"]["synthesis"] is True
    assert selected["validation"]["cosim"] is None
    assert original["validation"]["cosim"] is None


def test_real_baseline_cosim_evidence_is_preserved(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(archive, "REPO_ROOT", tmp_path)

    baseline = tmp_path / "baseline.cpp"
    baseline.write_text("int kernel(int x) { return x; }\n", encoding="utf-8")

    config = tmp_path / "task.json"
    config.write_text(
        json.dumps(
            {
                "output_dir": "out",
                "baseline": {
                    "source": "baseline.cpp",
                    "verification": {
                        "csim_passed": True,
                        "synthesis_passed": True,
                        "cosim_required": True,
                        "cosim_passed": True,
                    },
                },
                "requires_cosim": True,
                "selection": {"mode": "research_pareto"},
            }
        ),
        encoding="utf-8",
    )

    enriched = archive.preserve_candidate_state(
        config,
        {
            "baseline_metrics": _metrics(),
            "frequency_requirement": {
                "minimum_frequency_mhz": 100.0,
                "baseline_meets_requirement": True,
            },
            "resource_limits": {},
            "candidates": [],
            "pareto_archive": [],
        },
    )

    selected = enriched["candidate_state"]["selected_design"]
    assert selected["fully_verified"] is True
    assert selected["validation"]["cosim"] is True
