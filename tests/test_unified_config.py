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


@pytest.fixture
def valid_ppa_manifest() -> dict[str, Any]:
    return json.loads(
        Path("configs/tasks/vector_add_track_a.json").read_text(encoding="utf-8")
    )


def test_vector_add_ppa_manifest_loads() -> None:
    task = load_task(Path("configs/tasks/vector_add_track_a.json"))
    assert task.task_id == "hls_eval_vector_add_001"
    assert task.adapter_kind == "autonomous_ppa"
    assert set(task.data["adapter"]) == {"kind"}
    assert task.data["optimisation"]["target_loop_label"] == "vector_add_loop"
    assert task.data["target"]["minimum_frequency_mhz"] == 100.0
    assert "initialise_command" not in task.data["adapter"]
    assert "iteration_command" not in task.data["adapter"]


def test_vector_add_repair_manifest_loads() -> None:
    task = load_task(Path("configs/tasks/vector_add_repair.json"))
    assert task.task_id == "vector_add_functional_repair_001"
    assert task.adapter_kind == "direct_api_repair"
    assert set(task.data["adapter"]) == {"kind"}
    assert task.data["repair"]["editable_files"] == ["src/vector_add.cpp"]
    assert task.data["model"]["name"] == "Qwen/Qwen3.5-122B-A10B"
    assert task.data["target"]["minimum_frequency_mhz"] == 100.0


def test_u55c_platform_configuration_is_valid(
    valid_ppa_manifest: dict[str, Any],
) -> None:
    data = copy.deepcopy(valid_ppa_manifest)
    data["target"]["platform"] = "xilinx_u55c_gen3x16_xdma_3_202210_1"
    data["target"]["part"] = ""

    validate_task(data)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda data: data["artifacts"].update({"source": ""}),
            "artifacts.source must be a non-empty string",
        ),
        (
            lambda data: data["artifacts"].update({"testbench": []}),
            "artifacts.testbench must be a non-empty list of strings",
        ),
        (
            lambda data: data["interface"].update({"top_function": " "}),
            "interface.top_function must be a non-empty string",
        ),
        (
            lambda data: data["target"].update({"clock_period_ns": 0}),
            "target.clock_period_ns must be a positive number",
        ),
        (
            lambda data: data["target"].update({"clock_period_ns": -1}),
            "target.clock_period_ns must be a positive number",
        ),
        (
            lambda data: data["target"].update({"minimum_frequency_mhz": 0}),
            "target.minimum_frequency_mhz must be a positive number",
        ),
        (
            lambda data: data["target"].update({"minimum_frequency_mhz": -1}),
            "target.minimum_frequency_mhz must be a positive number",
        ),
        (
            lambda data: (
                data["target"].update({"part": "", "platform": ""})
            ),
            "target must define a non-empty platform or part",
        ),
        (
            lambda data: data["adapter"].update({"kind": "unsupported"}),
            "Unsupported adapter kind: unsupported",
        ),
        (
            lambda data: data["budgets"].update({"max_iterations": 0}),
            "budgets.max_iterations must be greater than zero",
        ),
        (
            lambda data: data["budgets"].update({"max_model_calls": -1}),
            "budgets.max_model_calls must be a non-negative integer",
        ),
        (
            lambda data: data["budgets"].update({"max_csim_calls": "1"}),
            "budgets.max_csim_calls must be a non-negative integer",
        ),
        (
            lambda data: data["budgets"].update({"max_csim_calls": True}),
            "budgets.max_csim_calls must be a non-negative integer",
        ),
        (
            lambda data: data["budgets"].update({"max_total_tokens": 0}),
            "budgets.max_total_tokens must be null or a positive integer",
        ),
    ],
    ids=[
        "missing_source",
        "missing_testbench",
        "missing_top_function",
        "zero_clock",
        "negative_clock",
        "zero_minimum_frequency",
        "negative_minimum_frequency",
        "missing_platform_and_part",
        "unsupported_adapter",
        "zero_iterations",
        "negative_budget",
        "non_integer_budget",
        "boolean_budget",
        "zero_token_budget",
    ],
)
def test_invalid_common_manifest_is_rejected(
    valid_ppa_manifest: dict[str, Any],
    mutate: Callable[[dict[str, Any]], Any],
    expected_error: str,
) -> None:
    data = copy.deepcopy(valid_ppa_manifest)
    mutate(data)

    with pytest.raises(ValueError, match=expected_error):
        validate_task(data)


def test_missing_budget_field_is_rejected(
    valid_repair_manifest: dict[str, Any],
) -> None:
    data = copy.deepcopy(valid_repair_manifest)
    data["budgets"].pop("max_model_calls")

    with pytest.raises(ValueError, match="Missing budget fields"):
        validate_task(data)


def test_direct_repair_external_config_is_rejected(
    valid_repair_manifest: dict[str, Any],
) -> None:
    data = copy.deepcopy(valid_repair_manifest)
    data["adapter"]["config"] = "legacy.json"

    with pytest.raises(
        ValueError,
        match="configured directly in the task manifest",
    ):
        validate_task(data)


def test_autonomous_ppa_external_config_is_rejected(
    valid_ppa_manifest: dict[str, Any],
) -> None:
    data = copy.deepcopy(valid_ppa_manifest)
    data["adapter"]["config"] = "configs/vector_add_ppa.json"

    with pytest.raises(
        ValueError,
        match="configured directly in the task manifest",
    ):
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
    ("artifact", "missing_path"),
    [
        ("source", "missing/source.cpp"),
        ("testbench", "missing/testbench.cpp"),
    ],
    ids=["missing_source_file", "missing_testbench_file"],
)
def test_missing_artifact_paths_are_rejected(
    tmp_path: Path,
    valid_ppa_manifest: dict[str, Any],
    artifact: str,
    missing_path: str,
) -> None:
    data = copy.deepcopy(valid_ppa_manifest)
    if artifact == "source":
        data["artifacts"]["source"] = missing_path
    else:
        data["artifacts"]["testbench"] = [missing_path]

    manifest_path = tmp_path / f"missing_{artifact}.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=f"task file does not exist: {missing_path}",
    ):
        load_task(manifest_path)


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
