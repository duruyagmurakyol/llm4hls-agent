from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from agent.config import TaskManifest, load_task
from agent.optimise.config_source import ppa_config_from_task


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = REPO_ROOT / "configs/tasks/vector_add_track_a.json"
BASELINE_TCL = (
    REPO_ROOT
    / "benchmarks/hls_eval/vector_add/scripts/run_agent_baseline.tcl"
)
U55C_PART = "xcu55c-fsvh2892-2L-e"


def test_vector_add_track_a_targets_u55c() -> None:
    task = load_task(TASK_PATH)
    target = task.data["target"]

    assert target["tool_version"] == "2025.2"
    assert target["part"] == U55C_PART
    assert target["clock_period_ns"] == 10.0
    assert target["minimum_frequency_mhz"] == 100.0


def test_vector_add_baseline_tcl_targets_same_u55c_part() -> None:
    source = BASELINE_TCL.read_text(encoding="utf-8")

    assert f"set_part {U55C_PART}" in source
    assert "xczu3eg" not in source.lower()


def test_task_specific_resource_limits_are_forwarded_to_ppa() -> None:
    task = load_task(TASK_PATH)
    data = deepcopy(task.data)
    expected = {
        "lut": 12345,
        "ff": 23456,
        "dsp": 321,
        "bram": 123,
    }
    data["target"]["resource_limits"] = expected

    synthetic = TaskManifest(path=task.path, data=data)
    config = ppa_config_from_task(synthetic)

    assert config["resource_limits"] == expected
