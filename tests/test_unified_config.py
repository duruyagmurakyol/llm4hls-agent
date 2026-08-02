from pathlib import Path

import pytest

from agent.config import load_task, validate_task


def test_vector_add_ppa_manifest_loads() -> None:
    task = load_task(Path("configs/tasks/vector_add_track_a.json"))
    assert task.task_id == "hls_eval_vector_add_001"
    assert task.adapter_kind == "legacy_ppa"


def test_vector_add_repair_manifest_loads() -> None:
    task = load_task(Path("configs/tasks/vector_add_repair.json"))
    assert task.task_id == "vector_add_functional_repair_001"
    assert task.adapter_kind == "direct_api_repair"


def test_missing_budget_is_rejected() -> None:
    data = {
        "task_id": "broken",
        "task_kind": "functional_failure",
        "artifacts": {"source": "a.cpp", "testbench": ["tb.cpp"]},
        "interface": {"top_function": "top"},
        "target": {},
        "budgets": {
            "max_iterations": 1,
            "max_csim_calls": 1,
            "max_cosim_calls": 0,
            "max_synthesis_calls": 0
        },
        "model": {},
        "adapter": {"kind": "direct_api_repair"},
        "output_dir": "results/test"
    }
    with pytest.raises(ValueError, match="Missing budget fields"):
        validate_task(data)
