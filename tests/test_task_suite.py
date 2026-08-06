from __future__ import annotations

import json
from pathlib import Path

from scripts.run_task_suite import TaskSpec, _filter_tasks, _result_row


def _task(tmp_path: Path, task_id: str, *, tag: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        path=str(tmp_path / task_id),
        output_dir=str(tmp_path / f"output_{task_id}"),
        collection="collection",
        source="source",
        tags=(tag,),
        timeout_seconds=60,
        max_agent_steps=None,
        resume=False,
        preflight=True,
    )


def test_suite_filters_by_id_path_collection_source_and_tags(tmp_path: Path) -> None:
    tasks = [
        _task(tmp_path, "track_a_projection", tag="repair"),
        _task(tmp_path, "auto_atax_001", tag="hls-eval"),
        _task(tmp_path, "auto_bicg_001", tag="hls-eval"),
    ]

    selected = _filter_tasks(
        tasks,
        only=["hls-eval"],
        skip=["*bicg*"],
        maximum=None,
    )

    assert [task.task_id for task in selected] == ["auto_atax_001"]


def test_suite_result_row_extracts_verified_hardware_and_budget(tmp_path: Path) -> None:
    task = _task(tmp_path, "track_a_residual", tag="cosim")
    output = task.resolved_output_dir
    output.mkdir(parents=True)
    (output / "unified_agent_result.json").write_text(
        json.dumps(
            {
                "success": True,
                "status": "completed_budget",
                "termination_reason": "verified_result_selected_after_budget_exhaustion",
                "selection_mode": "research_pareto",
                "final_design_verified": True,
                "meets_submission_frequency": True,
                "reference_harness_score_estimate": 3.1,
                "selected_design": {
                    "candidate_index": 0,
                    "candidate_file": "active_baseline.cpp",
                    "metrics": {
                        "frequency_mhz": 357.65,
                        "latency_ns": 190.128,
                        "throughput_period_ns": 178.944,
                        "resources_lut_used": 406,
                        "resources_ff_used": 231,
                        "resources_dsp_used": 0,
                        "resources_bram_used": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (output / "budget_summary.json").write_text(
        json.dumps(
            {
                "consumed": {
                    "total_tokens": 1200,
                    "csim_calls": 3,
                    "synthesis_calls": 3,
                    "cosim_calls": 2,
                },
                "track_a": {
                    "credits_spent": 55,
                    "credits_remaining": 25,
                },
            }
        ),
        encoding="utf-8",
    )

    row = _result_row(
        task,
        suite_run_id="run",
        started_at="2026-08-06T00:00:00+00:00",
        finished_at="2026-08-06T01:00:00+00:00",
        elapsed_seconds=3600,
        exit_code=0,
        timed_out=False,
        preflight="passed",
        log_path=tmp_path / "task.log",
    )

    assert row["success"] is True
    assert row["final_design_verified"] is True
    assert row["estimated_frequency_mhz"] == 357.65
    assert row["latency_ns"] == 190.128
    assert row["lut"] == 406
    assert row["total_tokens"] == 1200
    assert row["cosim_calls"] == 2
    assert row["reference_harness_credits_spent"] == 55
    assert row["reference_harness_credits_remaining"] == 25
