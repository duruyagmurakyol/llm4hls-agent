from pathlib import Path

from agent.config import TaskManifest
from agent.onboarding_safe import resolve_benchmark
from agent.optimise.config_source import ppa_config_from_task
from agent.optimise.runner import run_optimisation


def test_directory_resolution_is_in_memory(monkeypatch) -> None:
    def reject_write(*args, **kwargs):
        raise AssertionError("directory resolution must not write configuration files")

    monkeypatch.setattr(Path, "write_text", reject_write)

    task = resolve_benchmark(
        Path("benchmarks/vector_add/faults/functional_subtraction")
    )

    assert isinstance(task, TaskManifest)
    assert task.adapter_kind == "auto"
    assert task.data["task_kind"] == "unknown"
    assert task.data["artifacts"]["source"].endswith("src/vector_add.cpp")
    assert task.data["artifacts"]["testbench"][0].endswith(
        "testbench/vector_add_test.cpp"
    )
    assert task.data["interface"]["top_function"] == "vector_add"
    assert task.data["target"]["part"] == "xcu55c-fsvh2892-2L-e"
    assert task.data["target"]["clock_period_ns"] == 10.0
    assert task.data["repair"]["editable_files"] == ["src/vector_add.cpp"]
    assert "config" not in task.data["adapter"]


def test_optimisation_accepts_task_without_writing_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = resolve_benchmark(
        Path("benchmarks/vector_add/faults/functional_subtraction")
    )
    task.data["output_dir"] = str(tmp_path / "ppa")

    def reject_write(*args, **kwargs):
        raise AssertionError("PPA setup must not write an optimisation config")

    monkeypatch.setattr(Path, "write_text", reject_write)

    result = run_optimisation(task, status_only=True)
    config = ppa_config_from_task(task)

    assert result.status == "status_uninitialised"
    assert config["baseline"]["source"] == task.data["artifacts"]["source"]
    assert config["model"] == task.data["model"]
    assert config["budget"]["max_candidates"] == task.data["budgets"]["max_iterations"]
    assert config["budget"]["max_synthesis_calls"] == task.data["budgets"]["max_synthesis_calls"]
