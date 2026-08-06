from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import run_agent


def test_no_loop_target_retains_verified_baseline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_agent, "REPO_ROOT", tmp_path)
    output_dir = tmp_path / "experiments" / "track_a" / "projection_bugfix"
    output_dir.mkdir(parents=True)
    (output_dir / "verified_baseline.json").write_text(
        json.dumps(
            {
                "source": "experiments/track_a/projection_bugfix/active_baseline.cpp",
                "validation": {
                    "csim_passed": True,
                    "synthesis_passed": True,
                    "cosim_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    task = SimpleNamespace(
        task_id="track_a_projection_bugfix",
        output_dir="experiments/track_a/projection_bugfix",
    )

    fallback = run_agent._verified_baseline_fallback(
        task,
        ValueError("Could not map diagnosis target 'projection' to a source loop"),
    )

    assert fallback is not None
    result, result_path = fallback
    assert result.success is True
    assert result.status == "verified_baseline"
    assert result.termination_reason == "verified_baseline_no_mappable_ppa_target"
    assert result.to_dict()["selected_design"].endswith("active_baseline.cpp")
    assert result_path.is_file()


def test_unrelated_value_error_is_not_masked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_agent, "REPO_ROOT", tmp_path)
    task = SimpleNamespace(task_id="task", output_dir="out")
    assert run_agent._verified_baseline_fallback(task, ValueError("bad config")) is None
