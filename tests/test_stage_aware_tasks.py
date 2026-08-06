from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import TaskManifest
from agent.repair.prompt import build_strict_source_system_prompt
from agent.stage_aware import run_stage_aware_task, supports_stage_aware_task


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task(
    tmp_path: Path,
    *,
    kind: str,
    requires_cosim: bool,
) -> TaskManifest:
    root = tmp_path / kind
    root.mkdir()
    (root / "kernel.h").write_text(
        "void kernel(const int in[4], int out[4]);\n",
        encoding="utf-8",
    )
    (root / "kernel.cpp").write_text(
        '#include "kernel.h"\nvoid kernel(const int in[4], int out[4]) {\n'
        "  for (int i = 0; i < 4; ++i) out[i] = 0;\n}\n",
        encoding="utf-8",
    )
    (root / "tb.cpp").write_text(
        '#include "kernel.h"\nint main() { int a[4]={1,2,3,4}; int b[4]={}; '
        "kernel(a,b); return b[3] == 4 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    (root / "description.md").write_text(
        "Copy each input element to the corresponding output element.\n",
        encoding="utf-8",
    )
    (root / "task.cfg").write_text("[hls]\n", encoding="utf-8")
    return TaskManifest(
        path=root / "task.toml",
        data={
            "task_id": f"track_a_{kind}",
            "task_kind": kind,
            "task_root": str(root),
            "artifacts": {
                "source": str(root / "kernel.cpp"),
                "testbench": [str(root / "tb.cpp")],
                "headers": [str(root / "kernel.h")],
                "build_files": [str(root / "task.cfg")],
            },
            "interface": {"top_function": "kernel"},
            "target": {
                "minimum_frequency_mhz": 100.0,
                "resource_limits": {},
            },
            "budgets": {
                "max_iterations": 3,
                "max_model_calls": 3,
                "max_csim_calls": 5,
                "max_synthesis_calls": 5,
                "max_cosim_calls": 4,
                "max_total_tokens": None,
                "requires_cosim": requires_cosim,
                "track_a_credit_budget": 120,
                "track_a_credit_costs": {
                    "csim": 1,
                    "synthesis": 4,
                    "cosim": 20,
                },
            },
            "model": {
                "name": "test-model",
                "temperature": 0.0,
                "max_tokens": 1024,
                "timeout_seconds": 30,
            },
            "repair": {
                "benchmark_source": str(root),
                "editable_files": ["kernel.cpp"],
                "protected_files": [
                    "kernel.h",
                    "tb.cpp",
                    "description.md",
                    "task.cfg",
                ],
                "context_files": ["kernel.h", "tb.cpp", "description.md"],
            },
            "adapter": {"kind": "auto"},
            "output_dir": str(tmp_path / "output" / kind),
            "track_a": {
                "requires_cosim": requires_cosim,
                "difficulty": 1,
            },
        },
    )


def _passed_report(candidate: Path, *, metrics: bool = False, project: Path | None = None):
    report = {
        "passed": True,
        "timed_out": False,
        "return_code": 0,
        "failure_class": "none",
        "evidence": [],
        "candidate_hash": _hash(candidate),
        "candidate_file": str(candidate),
        "log_path": "tool.log",
    }
    if metrics:
        assert project is not None
        report.update(
            {
                "project_dir": str(project),
                "top_function": "kernel",
                "metrics": {
                    "clock_period_ns": 4.0,
                    "frequency_mhz": 250.0,
                    "latency_worst_cycles": 8,
                    "latency_average_cycles": 8,
                    "latency_best_cycles": 8,
                    "latency_ns": 32.0,
                    "throughput_period_ns": 4.0,
                    "resources_lut_used": 20,
                    "resources_ff_used": 10,
                    "resources_dsp_used": 0,
                    "resources_bram_used": 0,
                },
            }
        )
    return report


def _synthesis_factory(tmp_path: Path):
    counter = {"value": 0}

    def make(candidate: Path):
        counter["value"] += 1
        project = tmp_path / f"project_{counter['value']}"
        report_dir = project / "solution1" / "syn" / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "kernel_csynth.xml").write_text("<report/>\n", encoding="utf-8")
        return _passed_report(candidate, metrics=True, project=project)

    return make


def _response(source: str):
    return SimpleNamespace(
        content=source,
        raw_response={"choices": []},
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        latency_seconds=0.1,
    )


def test_generation_prompt_contract_allows_complete_implementation() -> None:
    prompt = build_strict_source_system_prompt(mode="generate")
    assert "Implement the complete kernel" in prompt
    assert "smallest justified change" not in prompt
    assert "Do not leave TODOs" in prompt


def test_stage_aware_support_is_explicit(tmp_path: Path) -> None:
    assert supports_stage_aware_task(_task(tmp_path, kind="generate", requires_cosim=False))
    assert supports_stage_aware_task(_task(tmp_path, kind="synth_fix", requires_cosim=False))
    assert supports_stage_aware_task(_task(tmp_path, kind="structural", requires_cosim=True))


def test_structural_failure_evidence_drives_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(tmp_path, kind="structural", requires_cosim=True)
    synthesis = _synthesis_factory(tmp_path)
    prompts: list[str] = []
    cosim_calls = {"value": 0}

    monkeypatch.setattr(
        "agent.stage_aware.run_csim",
        lambda _task, candidate: _passed_report(candidate),
    )
    monkeypatch.setattr("agent.stage_aware.run_synthesis", lambda _task, candidate: synthesis(candidate))

    def cosim(_task, candidate):
        cosim_calls["value"] += 1
        if cosim_calls["value"] == 1:
            return {
                **_passed_report(candidate),
                "passed": False,
                "return_code": 1,
                "failure_class": "cosim_deadlock",
                "evidence": ["RTL simulation made no forward progress"],
            }
        return _passed_report(candidate)

    monkeypatch.setattr("agent.stage_aware.run_cosim", cosim)

    def generate(**kwargs):
        prompts.append(kwargs["user_prompt"])
        source = (
            '#include "kernel.h"\nvoid kernel(const int in[4], int out[4]) {\n'
            "  for (int i = 0; i < 4; ++i) out[i] = in[i];\n}\n"
        )
        return source, _response(source)

    monkeypatch.setattr("agent.stage_aware.generate_repair", generate)

    result = run_stage_aware_task(task)

    assert result.success is True
    assert "FAILED STAGE: cosim" in prompts[0]
    assert "RTL simulation made no forward progress" in prompts[0]
    assert cosim_calls["value"] == 2
    state = json.loads(
        (Path(task.output_dir) / "candidate_state.json").read_text(encoding="utf-8")
    )
    assert state["selected_design_fully_verified"] is True


def test_synthesis_failure_evidence_drives_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(tmp_path, kind="synth_fix", requires_cosim=False)
    synthesis = _synthesis_factory(tmp_path)
    synth_calls = {"value": 0}
    prompts: list[str] = []

    monkeypatch.setattr(
        "agent.stage_aware.run_csim",
        lambda _task, candidate: _passed_report(candidate),
    )

    def synth(_task, candidate):
        synth_calls["value"] += 1
        if synth_calls["value"] == 1:
            return {
                **_passed_report(candidate),
                "passed": False,
                "return_code": 1,
                "failure_class": "unsupported_dynamic_allocation",
                "evidence": ["dynamic allocation is not synthesizable"],
            }
        return synthesis(candidate)

    monkeypatch.setattr("agent.stage_aware.run_synthesis", synth)

    def generate(**kwargs):
        prompts.append(kwargs["user_prompt"])
        source = (
            '#include "kernel.h"\nvoid kernel(const int in[4], int out[4]) {\n'
            "  for (int i = 0; i < 4; ++i) out[i] = in[i];\n}\n"
        )
        return source, _response(source)

    monkeypatch.setattr("agent.stage_aware.generate_repair", generate)

    result = run_stage_aware_task(task)

    assert result.success is True
    assert "FAILED STAGE: synthesis" in prompts[0]
    assert "dynamic allocation is not synthesizable" in prompts[0]
    assert synth_calls["value"] == 2


def test_generation_uses_generation_mode_before_tool_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(tmp_path, kind="generate", requires_cosim=False)
    synthesis = _synthesis_factory(tmp_path)
    modes: list[str] = []
    calls: list[str] = []

    def generate(**kwargs):
        modes.append(kwargs["mode"])
        source = (
            '#include "kernel.h"\nvoid kernel(const int in[4], int out[4]) {\n'
            "  for (int i = 0; i < 4; ++i) out[i] = in[i];\n}\n"
        )
        return source, _response(source)

    def csim(_task, candidate):
        calls.append("csim")
        return _passed_report(candidate)

    def synth(_task, candidate):
        calls.append("synthesis")
        return synthesis(candidate)

    monkeypatch.setattr("agent.stage_aware.generate_repair", generate)
    monkeypatch.setattr("agent.stage_aware.run_csim", csim)
    monkeypatch.setattr("agent.stage_aware.run_synthesis", synth)

    result = run_stage_aware_task(task)

    assert result.success is True
    assert modes == ["generate"]
    assert calls == ["csim", "synthesis"]
