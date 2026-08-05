from __future__ import annotations

import json
from pathlib import Path

from agent.archive import preserve_candidate_state
from agent.optimise.runner import _finish


def _config(tmp_path: Path) -> Path:
    config = {
        "experiment_name": "best_candidate_test",
        "benchmark": "kernel",
        "top_function": "kernel",
        "baseline": {
            "source": "baseline.cpp",
            "tcl": "task.tcl",
            "project_dir": "baseline_project",
        },
        "selection": {},
        "output_dir": "output",
        "budget": {
            "max_candidates": 3,
            "max_synthesis_calls": 3,
            "max_cosim_calls": 3,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _metrics(latency: int, lut: int) -> dict[str, int | float]:
    return {
        "clock_period_ns": 2.0,
        "frequency_mhz": 500.0,
        "minimum_frequency_mhz": 100.0,
        "maximum_clock_period_ns": 10.0,
        "latency_best_cycles": latency,
        "latency_average_cycles": latency,
        "latency_worst_cycles": latency,
        "latency_ns": latency * 2.0,
        "interval_min_cycles": 1,
        "interval_max_cycles": 1,
        "throughput_period_ns": 2.0,
        "resources_lut_used": lut,
        "resources_ff_used": 20,
        "resources_dsp_used": 0,
        "resources_bram_used": 0,
    }


def _verified_record(index: int, candidate_file: str, latency: int, lut: int, verdict: str) -> dict:
    return {
        "candidate_index": index,
        "candidate_file": candidate_file,
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": True,
        "fully_verified": True,
        "metrics": _metrics(latency, lut),
        "meets_frequency_requirement": True,
        "meets_resource_limits": True,
        "resource_limit_compliance": {"configured": False, "passed": True},
        "cost": {"total_tokens": 10 if index else 0, "tool_calls": 3 if index else 0, "tool_seconds": 1.0 if index else 0.0},
        "verdict": verdict,
    }


def _baseline_record() -> dict:
    return _verified_record(0, "baseline.cpp", 20, 20, "baseline")


def test_later_failure_keeps_previous_verified_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.archive.REPO_ROOT", tmp_path)
    (tmp_path / "baseline.cpp").write_text("void kernel() { /* baseline */ }\n")
    output = tmp_path / "output"
    output.mkdir()
    (output / "candidate_001.cpp").write_text("void kernel() { /* best */ }\n")
    (output / "candidate_002.cpp").write_text("void kernel() { /* broken latest */ }\n")

    best = _verified_record(
        1,
        "output/candidate_001.cpp",
        10,
        20,
        "accept_dominates_baseline",
    )
    broken = {
        "candidate_index": 2,
        "candidate_file": "output/candidate_002.cpp",
        "static_validation": True,
        "csim": False,
        "synthesis": None,
        "cosim": None,
        "fully_verified": False,
        "metrics": {},
        "verdict": "reject_csim",
        "reason": "failed",
    }
    summary = {
        "experiment_name": "best_candidate_test",
        "benchmark": "kernel",
        "baseline_metrics": _metrics(20, 20),
        "baseline_record": _baseline_record(),
        "budget": {
            "max_candidates": 3,
            "max_synthesis_calls": 3,
            "max_cosim_calls": 3,
            "synthesis_calls_used": 1,
            "synthesis_calls_remaining": 2,
            "cosim_calls_used": 1,
            "cosim_calls_remaining": 2,
        },
        "candidates": [best, broken],
        "pareto_archive": [best],
    }

    enriched = preserve_candidate_state(_config(tmp_path), summary)
    state = enriched["candidate_state"]

    assert state["latest_candidate"]["candidate_index"] == 2
    assert state["best_correct_candidate"]["candidate_index"] == 1
    assert state["best_ppa_candidate"]["candidate_index"] == 1
    assert state["selected_design"]["candidate_index"] == 1
    assert state["selected_design_fully_verified"] is True
    selected = tmp_path / state["selected_design"]["archived_file"]
    assert "best" in selected.read_text()
    assert "broken latest" not in selected.read_text()
    assert len(state["pareto_archive"]) == 1

    result = _finish(
        False,
        "failed",
        "intentional_final_candidate_failure",
        enriched,
        [],
    )
    assert result.success is True
    assert result.status == "completed_with_fallback"
    assert result.termination_reason == "intentional_final_candidate_failure"
    assert result.trajectory[-1]["stage"] == "select_best"
    assert result.trajectory[-1]["selected_design"]["candidate_index"] == 1
    assert result.trajectory[-1]["selected_design_fully_verified"] is True


def test_baseline_is_selected_when_no_candidate_is_verified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.archive.REPO_ROOT", tmp_path)
    (tmp_path / "baseline.cpp").write_text("void kernel() { /* baseline */ }\n")
    output = tmp_path / "output"
    output.mkdir()
    (output / "candidate_001.cpp").write_text("void kernel() { /* invalid */ }\n")

    baseline = _baseline_record()
    summary = {
        "experiment_name": "best_candidate_test",
        "benchmark": "kernel",
        "baseline_metrics": _metrics(20, 20),
        "baseline_record": baseline,
        "budget": {
            "max_candidates": 3,
            "max_synthesis_calls": 3,
            "max_cosim_calls": 3,
            "synthesis_calls_used": 0,
            "synthesis_calls_remaining": 3,
            "cosim_calls_used": 0,
            "cosim_calls_remaining": 3,
        },
        "candidates": [
            {
                "candidate_index": 1,
                "candidate_file": "output/candidate_001.cpp",
                "static_validation": False,
                "csim": None,
                "synthesis": None,
                "cosim": None,
                "fully_verified": False,
                "metrics": {},
                "verdict": "reject_static",
                "reason": "failed",
            }
        ],
        "pareto_archive": [baseline],
    }

    state = preserve_candidate_state(_config(tmp_path), summary)["candidate_state"]

    assert state["latest_candidate"]["candidate_index"] == 1
    assert state["best_correct_candidate"]["candidate_index"] == 0
    assert state["best_ppa_candidate"]["candidate_index"] == 0
    assert state["selected_design"]["candidate_index"] == 0
    assert state["selected_design_fully_verified"] is True
    selected = tmp_path / state["selected_design"]["archived_file"]
    assert "baseline" in selected.read_text()


def test_original_baseline_is_not_overwritten_by_later_updates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.archive.REPO_ROOT", tmp_path)
    baseline = tmp_path / "baseline.cpp"
    baseline.write_text("void kernel() { /* original */ }\n")
    (tmp_path / "output").mkdir()
    baseline_record = _baseline_record()
    summary = {
        "experiment_name": "best_candidate_test",
        "benchmark": "kernel",
        "baseline_metrics": _metrics(20, 20),
        "baseline_record": baseline_record,
        "budget": {},
        "candidates": [],
        "pareto_archive": [baseline_record],
    }
    config = _config(tmp_path)

    first = preserve_candidate_state(config, summary)["original_baseline"]
    baseline.write_text("void kernel() { /* later baseline bytes */ }\n")
    second = preserve_candidate_state(config, summary)["original_baseline"]

    archived = tmp_path / second["archived_file"]
    assert first["candidate_hash"] == second["candidate_hash"]
    assert "original" in archived.read_text()
    assert "later baseline bytes" not in archived.read_text()
