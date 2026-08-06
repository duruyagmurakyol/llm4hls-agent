from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.config import TaskManifest
from agent.final_cosim import enforce_final_cosim_policy
from agent.state import AgentResult, TrajectoryEvent


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _record(path: Path, index: int, latency: int) -> dict[str, Any]:
    return {
        "role": "pareto_member",
        "candidate_index": index,
        "candidate_file": str(path),
        "archived_file": str(path),
        "metrics": {
            "latency_ns": float(latency),
            "throughput_period_ns": float(latency),
            "resources_lut_used": 10 + index,
            "resources_ff_used": 10 + index,
            "resources_dsp_used": 0,
            "resources_bram_used": 0,
        },
        "fully_verified": False,
        "meets_frequency_requirement": True,
        "meets_resource_limits": True,
        "resource_limit_compliance": {"configured": False, "passed": True},
        "cost": {"total_tokens": index, "tool_calls": 2, "tool_seconds": 1.0},
        "verdict": "keep_pareto_candidate",
        "validation": {
            "static_validation": True,
            "csim": True,
            "synthesis": True,
            "cosim": None,
        },
    }


def _task(tmp_path: Path) -> TaskManifest:
    return TaskManifest(
        path=tmp_path / "task.json",
        data={
            "task_id": "final_cosim_test",
            "output_dir": str(tmp_path / "output"),
            "adapter": {"kind": "auto"},
        },
    )


def _result(task: TaskManifest, selected: Path, pareto: list[Path]) -> AgentResult:
    return AgentResult(
        task_id=task.task_id,
        success=True,
        status="completed",
        termination_reason="max_agent_steps_reached",
        output_dir=str(task.output_dir),
        trajectory=[
            TrajectoryEvent(
                step=1,
                stage="select_best",
                status="passed",
                details={
                    "selection_mode": "research_pareto",
                    "selected_design": str(selected),
                    "selected_design_fully_verified": False,
                    "selected_design_frequency_compliant": True,
                    "selected_design_resource_compliant": True,
                    "best_ppa_candidate": str(selected),
                    "best_correct_candidate": str(selected),
                    "pareto_archive": [str(path) for path in pareto],
                },
            )
        ],
    )


def _prepare(tmp_path: Path) -> tuple[TaskManifest, Path, Path, Path, AgentResult]:
    task = _task(tmp_path)
    output_dir = Path(task.output_dir)
    selected = tmp_path / "selected.cpp"
    fallback = tmp_path / "fallback.cpp"
    baseline = tmp_path / "baseline.cpp"
    selected.write_text("int selected() { return 1; }\n", encoding="utf-8")
    fallback.write_text("int fallback() { return 2; }\n", encoding="utf-8")
    baseline.write_text("int baseline() { return 0; }\n", encoding="utf-8")

    selected_record = _record(selected, 1, 10)
    fallback_record = _record(fallback, 2, 12)
    state = {
        "schema_version": 4,
        "selection_policy": {"mode": "research_pareto"},
        "selected_design_fully_verified": False,
        "selected_design_frequency_compliant": True,
        "selected_design_resource_compliant": True,
        "original_baseline": None,
        "latest_candidate": fallback_record,
        "best_correct_candidate": selected_record,
        "best_ppa_candidate": selected_record,
        "selected_design": selected_record,
        "pareto_archive": [selected_record, fallback_record],
    }
    _write_json(output_dir / "candidate_state.json", state)
    _write_json(
        output_dir / "verified_baseline.json",
        {
            "source": str(baseline),
            "candidate_hash": None,
            "metrics": _record(baseline, 0, 20)["metrics"],
            "validation": {
                "csim_passed": True,
                "synthesis_passed": True,
                "cosim_passed": None,
                "cosim_required": False,
            },
        },
    )
    result = _result(task, selected, [selected, fallback])
    return task, selected, fallback, baseline, result


def _report(path: Path, passed: bool) -> dict[str, Any]:
    return {
        "passed": passed,
        "failure_class": "none" if passed else "cosim_mismatch",
        "timed_out": False,
        "return_code": 0 if passed else 1,
        "candidate_hash": path.name,
        "candidate_file": str(path),
        "log_path": str(path.with_suffix(".log")),
    }


def test_selected_success_stops_without_validating_lower_ranked_designs(
    tmp_path: Path,
) -> None:
    task, selected, _fallback, _baseline, result = _prepare(tmp_path)
    calls: list[str] = []

    def fake_cosim(_task: TaskManifest, path: Path) -> dict[str, Any]:
        calls.append(path.name)
        return _report(path, True)

    audit = enforce_final_cosim_policy(task, result, cosim_runner=fake_cosim)

    assert calls == [selected.name]
    assert audit["policy"] == "ranked_selected_then_fallback"
    assert audit["status"] == "passed"
    assert audit["fallback_used"] is False
    assert audit["candidates_audited"] == 1
    assert audit["candidates_skipped_after_success"] == 2


def test_selected_cosim_failure_falls_back_to_next_ranked_pareto(
    tmp_path: Path,
) -> None:
    task, selected, fallback, _baseline, result = _prepare(tmp_path)
    calls: list[str] = []

    def fake_cosim(_task: TaskManifest, path: Path) -> dict[str, Any]:
        calls.append(path.name)
        return _report(path, path != selected)

    audit = enforce_final_cosim_policy(task, result, cosim_runner=fake_cosim)
    state = json.loads(
        (Path(task.output_dir) / "candidate_state.json").read_text(encoding="utf-8")
    )

    assert calls == [selected.name, fallback.name]
    assert audit["status"] == "passed_with_fallback"
    assert audit["fallback_used"] is True
    assert audit["metered_agent_budget"] is False
    assert audit["verified_pareto_count"] == 1
    assert audit["candidates_skipped_after_success"] == 1
    assert result.success is True
    assert result.status == "completed_with_cosim_fallback"
    assert result.termination_reason == "selected_cosim_failed_fallback_verified"
    assert state["selected_design_fully_verified"] is True
    assert Path(state["selected_design"]["candidate_file"]).resolve() == fallback.resolve()
    assert len(state["pareto_archive"]) == 1
    assert Path(state["pareto_archive"][0]["candidate_file"]).resolve() == fallback.resolve()


def test_baseline_is_only_run_after_ranked_pareto_candidates_fail(
    tmp_path: Path,
) -> None:
    task, selected, fallback, baseline, result = _prepare(tmp_path)
    calls: list[str] = []

    def fake_cosim(_task: TaskManifest, path: Path) -> dict[str, Any]:
        calls.append(path.name)
        return _report(path, path == baseline)

    audit = enforce_final_cosim_policy(task, result, cosim_runner=fake_cosim)

    assert calls == [selected.name, fallback.name, baseline.name]
    assert audit["status"] == "passed_with_fallback"
    assert audit["fallback_used"] is True
    assert audit["verified_pareto_count"] == 0


def test_all_cosim_failures_make_final_result_unsuccessful(tmp_path: Path) -> None:
    task, selected, fallback, baseline, result = _prepare(tmp_path)
    calls: list[str] = []

    def fake_cosim(_task: TaskManifest, path: Path) -> dict[str, Any]:
        calls.append(path.name)
        return _report(path, False)

    audit = enforce_final_cosim_policy(task, result, cosim_runner=fake_cosim)
    state = json.loads(
        (Path(task.output_dir) / "candidate_state.json").read_text(encoding="utf-8")
    )

    assert calls == [selected.name, fallback.name, baseline.name]
    assert audit["status"] == "failed"
    assert result.success is False
    assert result.status == "final_cosim_failed"
    assert result.termination_reason == "no_cosim_verified_pareto_or_baseline"
    assert state["selected_design"] is None
    assert state["pareto_archive"] == []


def test_existing_successful_selected_cosim_stops_without_tool_call(
    tmp_path: Path,
) -> None:
    task, selected, _fallback, _baseline, result = _prepare(tmp_path)
    state_path = Path(task.output_dir) / "candidate_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for record in [state["selected_design"], *state["pareto_archive"]]:
        if Path(record["candidate_file"]).resolve() == selected.resolve():
            record["validation"]["cosim"] = True
            record["fully_verified"] = True
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    calls: list[str] = []

    def fake_cosim(_task: TaskManifest, path: Path) -> dict[str, Any]:
        calls.append(path.name)
        return _report(path, True)

    audit = enforce_final_cosim_policy(task, result, cosim_runner=fake_cosim)

    assert calls == []
    assert audit["status"] == "passed"
    assert audit["fallback_used"] is False
    assert audit["candidates_audited"] == 1
    assert audit["candidates"][0]["reused_existing_cosim"] is True
