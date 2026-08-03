from pathlib import Path

from agent.config import TaskManifest
from agent.onboarding_safe import resolve_benchmark


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
    assert task.data["target"]["part"] == "xczu3eg-sfvc784-2-e"
    assert task.data["target"]["clock_period_ns"] == 10.0
    assert task.data["repair"]["editable_files"] == ["src/vector_add.cpp"]
    assert "config" not in task.data["adapter"]
