from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.budget import BudgetState
from agent.config import TaskManifest
from agent.controller import _run_direct_api_repair
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
        "part=xcu55c-fsvh2892-2L-e\n"
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
        ("ERROR: failed to compile RTL wrapper\n", False, "syntax_or_compile"),
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
    assert report["failure_class"] == "tool_report_missing"
    assert report["reports"] == []


def test_repair_runs_synthesis_then_cosim(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "results" / "run"
    candidate = run_dir / "workspace/src/kernel.cpp"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("void kernel() {}\n", encoding="utf-8")
    task = TaskManifest(
        path=tmp_path / "task.json",
        data={
            "task_id": "repair_then_cosim",
            "artifacts": {"build_files": ["task.cfg"]},
            "repair": {
                "benchmark_source": "benchmark",
                "editable_files": ["src/kernel.cpp"],
                "protected_files": ["src/kernel.h"],
                "host_validation": {"command": ["true"], "run_command": ["true"]},
                "independent_validation": {"enabled": True, "command": ["true"]},
            },
            "model": {"name": "model"},
            "output_dir": str(tmp_path / "output"),
        },
    )
    repair_result = {
        "experiment_id": "repair_then_cosim",
        "model": "model",
        "failure_class": "functional_mismatch",
        "tokens_used": 10,
        "input_tokens": 6,
        "output_tokens": 4,
        "modified_files": ["src/kernel.cpp"],
        "post_host_validation_passed": True,
        "independent_validation_passed": True,
    }
    supplied_candidates: list[Path] = []

    monkeypatch.setattr("agent.controller.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "agent.controller.run_repair",
        lambda *args, **kwargs: (True, run_dir, repair_result),
    )

    def fake_synthesis(supplied_task: TaskManifest, supplied_candidate: Path) -> dict[str, object]:
        supplied_candidates.append(supplied_candidate)
        return {
            "passed": True,
            "return_code": 0,
            "timed_out": False,
            "failure_class": "none",
            "evidence": [],
            "duration_seconds": 0.2,
            "log_path": "synthesis.log",
            "candidate_hash": "a" * 64,
            "metrics": {"latency_best_cycles": 4},
        }

    def fake_cosim(supplied_task: TaskManifest, supplied_candidate: Path) -> dict[str, object]:
        supplied_candidates.append(supplied_candidate)
        return {
            "passed": True,
            "return_code": 0,
            "timed_out": False,
            "failure_class": "none",
            "evidence": [],
            "duration_seconds": 0.3,
            "log_path": "cosim.log",
            "candidate_hash": "a" * 64,
            "reports": ["vector_add_cosim.rpt"],
        }

    monkeypatch.setattr("agent.controller.run_synthesis", fake_synthesis)
    monkeypatch.setattr("agent.controller.run_cosim", fake_cosim)
    budget = BudgetState(1, 1, 1, 1, 1, 100)

    result = _run_direct_api_repair(task, budget)

    assert result.success
    assert result.status == "fully_verified"
    assert result.termination_reason == "repair_synthesis_and_cosim_completed"
    assert [event.stage for event in result.trajectory] == [
        "repair",
        "post_repair_synthesis",
        "post_repair_cosim",
    ]
    assert supplied_candidates == [candidate, candidate]
    assert budget.synthesis_calls_used == 1
    assert budget.cosim_calls_used == 1
    assert budget.events[-1]["success"] is True
