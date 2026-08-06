from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.config import TaskManifest
from agent.optimise.config_source import ppa_config_from_task
from agent.track_a_selection import (
    OFFICIAL_TRACK_A_MODE,
    RESEARCH_PARETO_MODE,
    select_official_track_a,
)


def _record(
    index: int,
    latency: int,
    *,
    credits: int = 5,
    tokens: int = 100,
    lut: int = 100,
    requires_cosim: bool = False,
    cosim: bool | None = None,
) -> dict[str, object]:
    return {
        "candidate_index": index,
        "candidate_file": f"candidate_{index:03d}.cpp",
        "fully_verified": True,
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim_required": requires_cosim,
        "cosim": cosim,
        "official_validation_credits": credits,
        "metrics": {
            "latency_worst_cycles": latency,
            "latency_average_cycles": latency,
            "clock_period_ns": 4.0,
            "resources_lut_used": lut,
            "resources_ff_used": 10,
            "resources_dsp_used": 0,
            "resources_bram_used": 0,
        },
        "cost": {
            "total_tokens": tokens,
            "tool_calls": 2,
            "tool_seconds": 1.0,
        },
        "resource_limit_compliance": {"configured": False, "passed": True},
    }


def _config() -> dict[str, object]:
    return {
        "selection": {"mode": OFFICIAL_TRACK_A_MODE},
        "track_a": {"difficulty": 3, "requires_cosim": False},
    }


def test_official_selector_maximises_public_score_before_resources(tmp_path) -> None:
    (tmp_path / "original_scoring_baseline.json").write_text(
        json.dumps({"official_latency_cycles": 100}),
        encoding="utf-8",
    )
    baseline = _record(0, 100, credits=0, tokens=0, lut=10)
    candidate_one = _record(1, 50, lut=10)
    candidate_two = _record(2, 40, lut=10000)

    annotated, selected = select_official_track_a(
        [baseline, candidate_one, candidate_two],
        config=_config(),
        output_dir=tmp_path,
    )

    assert len(annotated) == 3
    assert selected is not None
    assert selected["candidate_index"] == 2
    assert selected["track_a_selection"]["public_score_estimate"] > (
        annotated[1]["track_a_selection"]["public_score_estimate"]
    )


def test_official_selector_does_not_invent_original_latency(tmp_path) -> None:
    (tmp_path / "original_scoring_baseline.json").write_text(
        json.dumps({"candidate_hash": "original", "synthesis_passed": None}),
        encoding="utf-8",
    )
    baseline = _record(0, 90, credits=0, tokens=0)
    candidate = _record(1, 30)

    annotated, selected = select_official_track_a(
        [baseline, candidate],
        config=_config(),
        output_dir=tmp_path,
    )

    assert selected is not None
    assert selected["candidate_index"] == 1
    assert annotated[0]["track_a_selection"]["acceleration"] is None
    assert annotated[1]["track_a_selection"]["acceleration"] is None
    assert annotated[0]["track_a_selection"]["public_score_estimate"] == pytest.approx(2.1)
    assert annotated[1]["track_a_selection"]["public_score_estimate"] == pytest.approx(2.1)


def test_equal_score_and_latency_prefers_lower_weighted_credits(tmp_path) -> None:
    (tmp_path / "original_scoring_baseline.json").write_text(
        json.dumps({"official_latency_cycles": 100}),
        encoding="utf-8",
    )
    expensive = _record(1, 50, credits=25, tokens=1, lut=1)
    efficient = _record(2, 50, credits=5, tokens=1000, lut=1000)

    _, selected = select_official_track_a(
        [expensive, efficient],
        config=_config(),
        output_dir=tmp_path,
    )

    assert selected is not None
    assert selected["candidate_index"] == 2


def test_required_cosim_failure_is_ineligible(tmp_path) -> None:
    (tmp_path / "original_scoring_baseline.json").write_text(
        json.dumps({"official_latency_cycles": 100}),
        encoding="utf-8",
    )
    valid = _record(1, 60, requires_cosim=True, cosim=True, credits=25)
    invalid = _record(2, 10, requires_cosim=True, cosim=False, credits=25)

    _, selected = select_official_track_a(
        [valid, invalid],
        config=_config(),
        output_dir=tmp_path,
    )

    assert selected is not None
    assert selected["candidate_index"] == 1


def test_track_a_manifest_defaults_to_research_pareto_selection(tmp_path) -> None:
    task = TaskManifest(
        path=tmp_path / "task.toml",
        data={
            "task_id": "track_a_example",
            "task_root": str(tmp_path),
            "artifacts": {
                "source": str(tmp_path / "kernel.cpp"),
                "build_files": [str(tmp_path / "task.cfg")],
            },
            "interface": {
                "top_function": "kernel",
                "protected_contract": [],
            },
            "target": {
                "clock_period_ns": 5.0,
                "minimum_frequency_mhz": 100.0,
                "resource_limits": {},
            },
            "budgets": {
                "max_iterations": 2,
                "max_model_calls": 2,
                "max_csim_calls": 4,
                "max_cosim_calls": 2,
                "max_synthesis_calls": 4,
            },
            "model": {},
            "optimisation": {},
            "output_dir": str(tmp_path / "output"),
            "track_a": {
                "difficulty": 2,
                "requires_cosim": False,
            },
        },
    )

    config = ppa_config_from_task(task)

    assert config["selection"]["mode"] == RESEARCH_PARETO_MODE
    assert config["track_a"]["difficulty"] == 2


def test_track_a_manifest_can_explicitly_request_reference_score_selection(
    tmp_path,
) -> None:
    task = TaskManifest(
        path=tmp_path / "task.toml",
        data={
            "task_id": "track_a_reference_mode",
            "task_root": str(tmp_path),
            "artifacts": {
                "source": str(tmp_path / "kernel.cpp"),
                "build_files": [str(tmp_path / "task.cfg")],
            },
            "interface": {
                "top_function": "kernel",
                "protected_contract": [],
            },
            "target": {
                "clock_period_ns": 5.0,
                "minimum_frequency_mhz": 100.0,
                "resource_limits": {},
            },
            "budgets": {
                "max_iterations": 2,
                "max_model_calls": 2,
                "max_csim_calls": 4,
                "max_cosim_calls": 2,
                "max_synthesis_calls": 4,
            },
            "model": {},
            "optimisation": {
                "selection": {"mode": OFFICIAL_TRACK_A_MODE},
            },
            "output_dir": str(tmp_path / "output"),
            "track_a": {
                "difficulty": 2,
                "requires_cosim": False,
            },
        },
    )

    config = ppa_config_from_task(task)

    assert config["selection"]["mode"] == OFFICIAL_TRACK_A_MODE
