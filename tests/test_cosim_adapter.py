from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.config import TaskManifest
from agent.tools.command_runner import CommandResult
from agent.tools.cosim import run_cosim


def _test_task(tmp_path: Path) -> tuple[TaskManifest, Path]:
    benchmark = tmp_path / "benchmark"
    (benchmark / "src").mkdir(parents=True)
    (benchmark / "testbench").mkdir()
    (benchmark / "src" / "kernel.cpp").write_text("void kernel() {}\n", encoding="utf-8")
    (benchmark / "src" / "kernel.h").write_text("void kernel();\n", encoding="utf-8")
    (benchmark / "testbench" / "kernel_tb.cpp").write_text(
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    build_file = benchmark / "task.cfg"
    build_file.write_text(
        "[hls]\n"
        "syn.file=src/kernel.cpp\n"
        "syn.top=kernel\n"
        "tb.file=testbench/kernel_tb.cpp\n"
        "part=xczu3eg-sfvc784-2-e\n"
        "clock=10ns\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.cpp"
    candidate.write_text("void kernel() {}\n", encoding="utf-8")
    task = TaskManifest(
        path=tmp_path / "task.json",
        data={
            "task_id": "test_cosim",
            "artifacts": {"build_files": [str(build_file)]},
            "output_dir": str(tmp_path / "output"),
        },
    )
    return task, candidate


def _project_dir(tcl: Path) -> Path:
    match = re.search(r'open_project -reset "([^"]+)"', tcl.read_text(encoding="utf-8"))
    assert match is not None
    return Path(match.group(1))


def test_run_cosim_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    task, candidate = _test_task(tmp_path)

    def fake_vitis(tcl: Path, cwd: Path, log_path: Path, timeout_seconds: int) -> CommandResult:
        report_dir = _project_dir(tcl) / "solution1/sim/report"
        report_dir.mkdir(parents=True)
        (report_dir / "verilog.log").write_text("PASS\n", encoding="utf-8")
        output = "C/RTL co-simulation finished: PASS\n"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        return CommandResult(
            command=("vitis-run", "--mode", "hls", "--tcl", str(tcl)),
            return_code=0,
            output=output,
            cwd=str(cwd),
            timed_out=False,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=0.3,
        )

    monkeypatch.setattr("agent.tools.cosim._run_vitis", fake_vitis)
    report = run_cosim(task, candidate)

    assert report["passed"]
    assert report["failure_class"] == "none"
    assert report["return_code"] == 0
    assert report["duration_seconds"] == 0.3
    assert len(report["candidate_hash"]) == 64
    assert report["reports"]
    assert Path(report["reports"][0]).is_file()
    tcl_text = Path(report["generated_tcl"]).read_text(encoding="utf-8")
    assert tcl_text.index("csynth_design") < tcl_text.index("cosim_design")
    assert report["cosim_run"] is True
    assert report["baseline_modified"] is False


@pytest.mark.parametrize(
    ("output", "timed_out", "expected_class"),
    [
        ("FAIL index=0 expected=17 actual=-15\nSimulation failed\n", False, "cosim_mismatch"),
        ("ERROR: deadlock detected; no progress\n", False, "cosim_deadlock"),
        ("ERROR: failed to compile RTL wrapper\n", False, "cosim_compile"),
        ("TIMEOUT: command exceeded 900 seconds\n", True, "cosim_timeout"),
    ],
)
def test_run_cosim_classifies_failures(
    tmp_path: Path,
    monkeypatch,
    output: str,
    timed_out: bool,
    expected_class: str,
) -> None:
    task, candidate = _test_task(tmp_path)

    def fake_vitis(tcl: Path, cwd: Path, log_path: Path, timeout_seconds: int) -> CommandResult:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        return CommandResult(
            command=("vitis-run", "--mode", "hls", "--tcl", str(tcl)),
            return_code=-15 if timed_out else 1,
            output=output,
            cwd=str(cwd),
            timed_out=timed_out,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=0.3,
        )

    monkeypatch.setattr("agent.tools.cosim._run_vitis", fake_vitis)
    report = run_cosim(task, candidate)

    assert not report["passed"]
    assert report["failure_class"] == expected_class
    assert report["evidence"]


def test_run_cosim_requires_report(tmp_path: Path, monkeypatch) -> None:
    task, candidate = _test_task(tmp_path)

    def fake_vitis(tcl: Path, cwd: Path, log_path: Path, timeout_seconds: int) -> CommandResult:
        output = "C/RTL co-simulation finished\n"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        return CommandResult(
            command=("vitis-run", "--mode", "hls", "--tcl", str(tcl)),
            return_code=0,
            output=output,
            cwd=str(cwd),
            timed_out=False,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=0.3,
        )

    monkeypatch.setattr("agent.tools.cosim._run_vitis", fake_vitis)
    report = run_cosim(task, candidate)

    assert not report["passed"]
    assert report["failure_class"] == "missing_cosim_report"
    assert report["reports"] == []
