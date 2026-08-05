from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.config import TaskManifest
from agent.tools.command_runner import CommandResult
from agent.tools.synthesis import run_synthesis


def _task(tmp_path: Path) -> tuple[TaskManifest, Path]:
    benchmark = tmp_path / "benchmark"
    (benchmark / "src").mkdir(parents=True)
    (benchmark / "src/kernel.cpp").write_text("void kernel() {}\n", encoding="utf-8")
    (benchmark / "src/kernel.h").write_text("void kernel();\n", encoding="utf-8")
    cfg = benchmark / "task.cfg"
    cfg.write_text(
        "[hls]\n"
        "syn.file=src/kernel.cpp\n"
        "syn.top=kernel\n"
        "part=xczu3eg-sfvc784-2-e\n"
        "clock=10ns\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.cpp"
    candidate.write_text("void kernel() {}\n", encoding="utf-8")
    task = TaskManifest(
        path=tmp_path / "task.json",
        data={
            "task_id": "synthesis_taxonomy",
            "artifacts": {"build_files": [str(cfg)]},
            "output_dir": str(tmp_path / "output"),
        },
    )
    return task, candidate


def _project_dir(tcl: Path) -> Path:
    match = re.search(r'open_project -reset "([^"]+)"', tcl.read_text(encoding="utf-8"))
    assert match is not None
    return Path(match.group(1))


@pytest.mark.parametrize(
    ("output", "timed_out", "expected"),
    [
        (
            "ERROR: dynamic memory allocation is not supported for synthesis; design is not synthesizable",
            False,
            "synthesis_unsupported_construct",
        ),
        ("ERROR: top function not found: kernel", False, "top_function_mismatch"),
        ("kernel.cpp:9: error: expected ';' before '}'", False, "syntax_or_compile"),
        ("csynth_design timed out after 600 seconds", True, "synthesis_timeout"),
    ],
)
def test_run_synthesis_uses_canonical_failure_classes(
    tmp_path: Path,
    monkeypatch,
    output: str,
    timed_out: bool,
    expected: str,
) -> None:
    task, candidate = _task(tmp_path)

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
            elapsed_seconds=0.2,
        )

    monkeypatch.setattr("agent.tools.synthesis._run_vitis", fake_vitis)
    report = run_synthesis(task, candidate)

    assert report["passed"] is False
    assert report["failure_class"] == expected
    assert report["evidence"]


def test_run_synthesis_distinguishes_missing_report(tmp_path: Path, monkeypatch) -> None:
    task, candidate = _task(tmp_path)

    def fake_vitis(tcl: Path, cwd: Path, log_path: Path, timeout_seconds: int) -> CommandResult:
        output = "INFO: Finished Command csynth_design\n"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        assert _project_dir(tcl).is_absolute()
        return CommandResult(
            command=("vitis-run", "--mode", "hls", "--tcl", str(tcl)),
            return_code=0,
            output=output,
            cwd=str(cwd),
            timed_out=False,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=0.2,
        )

    monkeypatch.setattr("agent.tools.synthesis._run_vitis", fake_vitis)
    report = run_synthesis(task, candidate)

    assert report["passed"] is False
    assert report["failure_class"] == "tool_report_missing"
    assert report["evidence"] == ["Missing top synthesis report for kernel"]
