from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent import track_a_scoring
from agent.config import TaskManifest
from agent.state import AgentResult, TrajectoryEvent
from agent.track_a_scoring import (
    estimate_public_track_a_score,
    official_latency_cycles,
    write_track_a_score_estimate,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _task(tmp_path: Path) -> TaskManifest:
    source = tmp_path / "staging/kernel.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("void kernel() {}\n", encoding="utf-8")
    return TaskManifest(
        path=tmp_path / "task.toml",
        data={
            "task_id": "track_a_kernel",
            "task_kind": "optimize",
            "artifacts": {
                "source": str(source),
                "testbench": [str(tmp_path / "staging/kernel_tb.cpp")],
                "build_files": [str(tmp_path / "staging/task.cfg")],
            },
            "interface": {"top_function": "kernel"},
            "target": {
                "part": "xcu55c-fsvh2892-2L-e",
                "clock_period_ns": 5.0,
            },
            "budgets": {},
            "model": {},
            "adapter": {"kind": "auto"},
            "output_dir": str(tmp_path / "output"),
            "track_a": {
                "difficulty": 2,
                "requires_cosim": False,
            },
        },
    )


def test_official_latency_prefers_worst_then_average() -> None:
    assert official_latency_cycles(
        {
            "latency_best_cycles": 10,
            "latency_average_cycles": 20,
            "latency_worst_cycles": 30,
        }
    ) == 30
    assert official_latency_cycles(
        {
            "latency_best_cycles": 10,
            "latency_average_cycles": 20,
            "latency_worst_cycles": None,
        }
    ) == 20
    assert official_latency_cycles({"latency_best_cycles": 10}) is None
    assert official_latency_cycles({"latency_worst_cycles": 0}) == 0


def test_public_score_uses_acceleration_and_caps_at_eight() -> None:
    result = estimate_public_track_a_score(
        difficulty=2,
        public_correct=True,
        synthesis_passed=True,
        original_latency_cycles=1000,
        candidate_latency_cycles=500,
    )
    assert result["acceleration"] == 2.0
    assert result["ppa_norm"] == 0.25
    assert result["public_score_estimate"] == pytest.approx(1.55)

    capped = estimate_public_track_a_score(
        difficulty=2,
        public_correct=True,
        synthesis_passed=True,
        original_latency_cycles=1000,
        candidate_latency_cycles=100,
    )
    assert capped["acceleration"] == 10.0
    assert capped["ppa_norm"] == 1.0
    assert capped["public_score_estimate"] == pytest.approx(2.0)

    failed = estimate_public_track_a_score(
        difficulty=4,
        public_correct=False,
        synthesis_passed=True,
        original_latency_cycles=1000,
        candidate_latency_cycles=100,
    )
    assert failed["public_score_estimate"] == 0.0


def test_zero_latency_has_no_acceleration_component() -> None:
    result = estimate_public_track_a_score(
        difficulty=2,
        public_correct=True,
        synthesis_passed=True,
        original_latency_cycles=0,
        candidate_latency_cycles=0,
    )
    assert result["acceleration"] is None
    assert result["ppa_norm"] == 0.0
    assert result["public_score_estimate"] == pytest.approx(1.4)


def test_score_report_reuses_existing_synthesis_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(track_a_scoring, "REPO_ROOT", tmp_path)
    task = _task(tmp_path)
    source = Path(task.data["artifacts"]["source"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output_dir = Path(task.output_dir)

    original_project = tmp_path / "original_project"
    original_report = (
        original_project / "solution1/syn/report/kernel_csynth.xml"
    )
    original_report.parent.mkdir(parents=True)
    original_report.write_text("<Report/>\n", encoding="utf-8")
    _write_json(
        output_dir / "synthesis" / digest[:12] / "result.json",
        {
            "passed": True,
            "candidate_hash": digest,
            "project_dir": str(original_project),
            "metrics": {
                "clock_period_ns": 6.0,
                "latency_best_cycles": 80,
                "latency_average_cycles": 90,
                "latency_worst_cycles": 100,
            },
        },
    )

    selected = output_dir / "active_baseline.cpp"
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text("void kernel() { int x = 1; }\n", encoding="utf-8")
    selected_hash = hashlib.sha256(selected.read_bytes()).hexdigest()
    _write_json(
        output_dir / "verified_baseline.json",
        {
            "source": str(selected),
            "candidate_hash": selected_hash,
            "metrics": {
                "clock_period_ns": 4.0,
                "latency_best_cycles": 40,
                "latency_average_cycles": 45,
                "latency_worst_cycles": 50,
            },
            "validation": {
                "csim_passed": True,
                "synthesis_passed": True,
                "cosim_required": False,
                "cosim_passed": None,
            },
        },
    )
    result = AgentResult(
        task_id=task.task_id,
        success=True,
        status="verified_baseline",
        termination_reason="complete",
        output_dir=str(task.output_dir),
        trajectory=[
            TrajectoryEvent(
                step=1,
                stage="select_best",
                status="passed",
                details={"selected_design": str(selected)},
            )
        ],
    )

    scored = write_track_a_score_estimate(task, result)
    assert scored is not None
    path, report = scored

    assert path == output_dir / "track_a_score_estimate.json"
    assert report["hidden_tests_used"] is False
    assert report["public_correct"] is True
    assert report["original_scoring_baseline"]["official_latency_cycles"] == 100
    assert report["selected_design"]["official_latency_cycles"] == 50
    assert report["selected_design"]["target_frequency_mhz"] == 200.0
    assert report["selected_design"]["estimated_frequency_mhz"] == 250.0
    assert report["selected_design"]["meets_target_clock"] is True
    assert report["acceleration"] == 2.0
    assert report["public_score_estimate"] == pytest.approx(1.55)

    original = json.loads(
        (output_dir / "original_scoring_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert original["candidate_hash"] == digest
    assert original["synthesis_passed"] is True
    assert original["official_latency_cycles"] == 100
    top_report = Path(str(original["top_csynth_xml"]))
    if not top_report.is_absolute():
        top_report = tmp_path / top_report
    assert top_report.is_file()
