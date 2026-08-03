from pathlib import Path

import pytest

from agent.config import load_task, validate_task


def test_vector_add_ppa_manifest_loads() -> None:
    task = load_task(Path("configs/tasks/vector_add_track_a.json"))
    assert task.task_id == "hls_eval_vector_add_001"
    assert task.adapter_kind == "autonomous_ppa"
    assert task.data["adapter"]["config"] == "configs/vector_add_ppa.json"
    assert "initialise_command" not in task.data["adapter"]
    assert "iteration_command" not in task.data["adapter"]


def test_vector_add_repair_manifest_loads() -> None:
    task = load_task(Path("configs/tasks/vector_add_repair.json"))
    assert task.task_id == "vector_add_functional_repair_001"
    assert task.adapter_kind == "direct_api_repair"
    assert set(task.data["adapter"]) == {"kind"}
    assert task.data["repair"]["editable_files"] == ["src/vector_add.cpp"]
    assert task.data["model"]["name"] == "Qwen/Qwen3.5-122B-A10B"


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


def test_direct_repair_external_config_is_rejected() -> None:
    data = {
        "task_id": "repair",
        "task_kind": "functional_failure",
        "artifacts": {"source": "a.cpp", "testbench": ["tb.cpp"]},
        "interface": {"top_function": "top"},
        "target": {},
        "budgets": {
            "max_iterations": 1,
            "max_csim_calls": 1,
            "max_cosim_calls": 0,
            "max_synthesis_calls": 0,
            "max_model_calls": 1
        },
        "model": {},
        "repair": {},
        "adapter": {"kind": "direct_api_repair", "config": "legacy.json"},
        "output_dir": "results/test"
    }
    with pytest.raises(ValueError, match="configured directly in the task manifest"):
        validate_task(data)
