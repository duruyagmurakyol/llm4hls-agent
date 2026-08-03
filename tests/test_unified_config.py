import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from agent.config import load_task, validate_task


@pytest.fixture
def valid_repair_manifest() -> dict[str, Any]:
    return json.loads(
        Path("configs/tasks/vector_add_repair.json").read_text(encoding="utf-8")
    )


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


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda data: data.pop("repair"),
            "repair must be an object",
        ),
        (
            lambda data: data["repair"]["protected_files"].append(
                data["repair"]["editable_files"][0]
            ),
            "repair editable and protected files overlap",
        ),
        (
            lambda data: data["repair"]["host_validation"].update(
                {"command": []}
            ),
            "repair.host_validation.command must be a non-empty list of strings",
        ),
    ],
    ids=[
        "missing_repair_section",
        "editable_protected_overlap",
        "empty_host_command",
    ],
)
def test_invalid_repair_manifest_is_rejected(
    valid_repair_manifest: dict[str, Any],
    mutate: Callable[[dict[str, Any]], Any],
    expected_error: str,
) -> None:
    data = copy.deepcopy(valid_repair_manifest)
    mutate(data)

    with pytest.raises(ValueError, match=expected_error):
        validate_task(data)


@pytest.mark.parametrize(
    "case_name",
    [
        "missing_benchmark_directory",
        "missing_repair_file",
    ],
)
def test_invalid_repair_paths_are_rejected(
    tmp_path: Path,
    valid_repair_manifest: dict[str, Any],
    case_name: str,
) -> None:
    data = copy.deepcopy(valid_repair_manifest)

    if case_name == "missing_benchmark_directory":
        data["repair"]["benchmark_source"] = str(tmp_path / "missing")
        expected_error = "repair.benchmark_source does not exist"
    else:
        benchmark = tmp_path / "benchmark"
        (benchmark / "src").mkdir(parents=True)
        (benchmark / "testbench").mkdir()
        (benchmark / "src" / "vector_add.h").write_text("", encoding="utf-8")
        (benchmark / "testbench" / "vector_add_test.cpp").write_text(
            "", encoding="utf-8"
        )
        (benchmark / "task.cfg").write_text("", encoding="utf-8")
        data["repair"]["benchmark_source"] = str(benchmark)
        expected_error = "repair file does not exist: src/vector_add.cpp"

    manifest_path = tmp_path / f"{case_name}.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        load_task(manifest_path)
