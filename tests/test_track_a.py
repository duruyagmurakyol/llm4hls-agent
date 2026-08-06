from __future__ import annotations

from pathlib import Path

from agent import config, track_a
from agent.optimise.config_source import ppa_config_from_task


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_track_a_package_stages_only_public_files(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    package = tmp_path / "projection_bugfix"
    repo_root.mkdir()
    package.mkdir()

    _write(
        package / "task.toml",
        """task_id = "projection_bugfix"
task_type = "repair"
difficulty = 2
top = "projection"
kernel_file = "projection.cpp"
header_files = ["projection.h"]
public_tb = "projection_tb.cpp"
budget = 20
initial_condition = "public functional fault"

[target]
part = "xcu55c-fsvh2892-2L-e"
clock_ns = 5.0
""",
    )
    _write(package / "description.md", "Repair the projection kernel.\n")
    _write(
        package / "projection.h",
        "void projection(int in[4], int out[4]);\n",
    )
    _write(
        package / "projection.cpp",
        '#include "projection.h"\nvoid projection(int in[4], int out[4]) { out[0] = in[0]; }\n',
    )
    _write(
        package / "projection_tb.cpp",
        '#include "projection.h"\nint main() { int a[4]={}; int b[4]={}; projection(a,b); return 0; }\n',
    )
    _write(package / "hidden" / "projection_tb.cpp", "SECRET_HIDDEN_TEST\n")
    _write(package / "reference" / "projection.cpp", "SECRET_REFERENCE\n")

    monkeypatch.setattr(track_a, "REPO_ROOT", repo_root)
    monkeypatch.setattr(config, "REPO_ROOT", repo_root)

    task = track_a.resolve_track_a_task(package)
    staged = repo_root / "experiments" / "track_a_staging" / "projection_bugfix"

    assert task.adapter_kind == "auto"
    assert task.data["task_kind"] == "repair"
    assert task.data["target"]["part"] == "xcu55c-fsvh2892-2L-e"
    assert task.data["target"]["clock_period_ns"] == 5.0
    assert task.data["track_a"]["hidden_and_reference_excluded"] is True
    assert task.data["track_a"]["requires_cosim"] is False
    assert task.data["budgets"]["track_a_credit_budget"] == 20
    assert task.data["budgets"]["track_a_credit_costs"] == {
        "csim": 1,
        "synthesis": 4,
        "cosim": 20,
    }
    assert ppa_config_from_task(task)["requires_cosim"] is False
    assert (staged / "projection.cpp").is_file()
    assert (staged / "projection.h").is_file()
    assert (staged / "projection_tb.cpp").is_file()
    assert (staged / "description.md").is_file()
    assert (staged / "task.toml").is_file()
    assert (staged / "task.cfg").is_file()
    assert not (staged / "hidden").exists()
    assert not (staged / "reference").exists()
    assert "SECRET_HIDDEN_TEST" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in staged.rglob("*")
        if path.is_file()
    )
    assert "SECRET_REFERENCE" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in staged.rglob("*")
        if path.is_file()
    )

    host_command = task.data["repair"]["host_validation"]["command"]
    assert host_command[:2] == ["bash", "-lc"]
    assert "command -v vitis-run" in host_command[2]
    assert '$vitis_root/include' in host_command[2]
    assert "projection.cpp projection_tb.cpp" in host_command[2]

    config.validate_task(task.data)
    config.validate_task_paths(task.data)


def test_structural_track_a_package_requires_cosim(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    package = tmp_path / "stream_task"
    repo_root.mkdir()
    package.mkdir()

    _write(
        package / "task.toml",
        """task_id = "stream_task"
task_type = "structural"
top = "kernel"
kernel_file = "kernel.cpp"
header_files = ["kernel.h"]
public_tb = "kernel_tb.cpp"
budget = 80
requires_cosim = true

[target]
part = "xcu55c-fsvh2892-2L-e"
clock_ns = 5.0
""",
    )
    _write(package / "kernel.h", "void kernel();\n")
    _write(package / "kernel.cpp", '#include "kernel.h"\nvoid kernel() {}\n')
    _write(package / "kernel_tb.cpp", '#include "kernel.h"\nint main(){kernel();}\n')

    monkeypatch.setattr(track_a, "REPO_ROOT", repo_root)
    monkeypatch.setattr(config, "REPO_ROOT", repo_root)

    task = track_a.resolve_track_a_task(package)
    assert task.data["track_a"]["requires_cosim"] is True
    assert task.data["budgets"]["track_a_credit_budget"] == 80
    assert ppa_config_from_task(task)["requires_cosim"] is True


def test_track_a_rejects_path_escape(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    package = tmp_path / "bad_task"
    repo_root.mkdir()
    package.mkdir()
    _write(tmp_path / "outside.cpp", "void bad() {}\n")
    _write(
        package / "task.toml",
        """top = "bad"
kernel_file = "../outside.cpp"
public_tb = "tb.cpp"
""",
    )
    _write(package / "tb.cpp", "int main() { return 0; }\n")

    monkeypatch.setattr(track_a, "REPO_ROOT", repo_root)
    monkeypatch.setattr(config, "REPO_ROOT", repo_root)

    try:
        track_a.resolve_track_a_task(package)
    except ValueError as error:
        assert "escapes the task directory" in str(error)
    else:
        raise AssertionError("path traversal was accepted")
