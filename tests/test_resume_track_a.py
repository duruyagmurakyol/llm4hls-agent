from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.config import TaskManifest
from agent import resume


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _baseline_record(root: Path, *, cosim_required: bool, cosim_passed: bool | None) -> dict[str, object]:
    source = root / "active_baseline.cpp"
    source.write_text("void kernel() {}\n", encoding="utf-8")
    project = root / "verified_baseline_project"
    report = project / "solution1/syn/report/kernel_csynth.xml"
    report.parent.mkdir(parents=True)
    report.write_text("<profile/>\n", encoding="utf-8")
    return {
        "source": str(source),
        "candidate_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
        "project_dir": str(project),
        "top_csynth_xml": str(report),
        "metrics": {"latency_worst_cycles": 10},
        "validation": {
            "csim_passed": True,
            "synthesis_passed": True,
            "cosim_required": cosim_required,
            "cosim_passed": cosim_passed,
        },
    }


def _task(root: Path) -> TaskManifest:
    return TaskManifest(
        path=root / "task.toml",
        data={
            "task_id": "track_a_resume",
            "adapter": {"kind": "auto"},
            "output_dir": str(root),
            "budgets": {
                "max_iterations": 5,
                "max_model_calls": 5,
                "max_csim_calls": 7,
                "max_cosim_calls": 4,
                "max_synthesis_calls": 6,
                "max_total_tokens": None,
                "track_a_credit_budget": 40,
                "track_a_credit_costs": {
                    "csim": 1,
                    "synthesis": 4,
                    "cosim": 20,
                },
            },
        },
    )


def test_resume_accepts_verified_baseline_when_cosim_is_not_required(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(resume, "REPO_ROOT", tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    _write_json(
        output / "verified_baseline.json",
        _baseline_record(output, cosim_required=False, cosim_passed=None),
    )

    record = resume.load_resumable_baseline(output)

    assert record["validation"]["cosim_required"] is False
    assert record["validation"]["cosim_passed"] is None


def test_resume_still_requires_cosim_when_task_record_requires_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(resume, "REPO_ROOT", tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    _write_json(
        output / "verified_baseline.json",
        _baseline_record(output, cosim_required=True, cosim_passed=None),
    )

    with pytest.raises(ValueError, match="not fully verified"):
        resume.load_resumable_baseline(output)


def test_resume_restores_cumulative_track_a_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(resume, "REPO_ROOT", tmp_path)
    task = _task(tmp_path / "run")
    _write_json(
        tmp_path / "run/budget_summary.json",
        {
            "consumed": {
                "iterations": 2,
                "model_calls": 2,
                "csim_calls": 1,
                "cosim_calls": 0,
                "synthesis_calls": 1,
                "input_tokens": 594,
                "output_tokens": 140,
                "total_tokens": 734,
            },
            "track_a": {
                "credit_budget": 40,
                "credit_costs": {
                    "csim": 1,
                    "synthesis": 4,
                    "cosim": 20,
                },
                "credits_spent": 5,
                "credits_remaining": 35,
            },
            "events": [{"resource": "csim_calls", "stage": "initial_csim"}],
        },
    )

    budget = resume._load_resumable_budget(task)

    assert budget.iterations_used == 2
    assert budget.model_calls_used == 2
    assert budget.csim_calls_used == 1
    assert budget.synthesis_calls_used == 1
    assert budget.total_tokens_used == 734
    assert budget.track_a_credits_used == 5
    assert budget.track_a_credits_remaining == 35
    assert budget.can_generate_candidate(reserve_csim=1, reserve_synthesis=1)
    assert budget.events[-1]["resource"] == "resume"


def test_resume_rejects_credit_usage_above_official_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(resume, "REPO_ROOT", tmp_path)
    task = _task(tmp_path / "run")
    _write_json(
        tmp_path / "run/budget_summary.json",
        {
            "consumed": {},
            "track_a": {
                "credit_costs": {
                    "csim": 1,
                    "synthesis": 4,
                    "cosim": 20,
                },
                "credits_spent": 41,
            },
        },
    )

    with pytest.raises(ValueError, match="exceeds the configured official budget"):
        resume._load_resumable_budget(task)
