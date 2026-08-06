from __future__ import annotations

import json
from pathlib import Path

from agent.state import AgentResult, TrajectoryEvent
from scripts.run_agent import _enrich_track_a_result


def test_verified_selection_converts_budget_stop_to_completed_result(
    tmp_path: Path,
) -> None:
    selected = {
        "candidate_index": 0,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "meets_resource_limits": True,
    }
    (tmp_path / "candidate_state.json").write_text(
        json.dumps(
            {
                "selection_policy": {"mode": "research_pareto"},
                "selected_design": selected,
                "selected_design_fully_verified": True,
                "selected_design_frequency_compliant": True,
                "selected_design_resource_compliant": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "budget_summary.json").write_text(
        json.dumps(
            {
                "track_a": {
                    "credit_budget": 80,
                    "credits_spent": 75,
                    "credits_remaining": 5,
                    "credit_costs": {
                        "csim": 1,
                        "synthesis": 4,
                        "cosim": 20,
                    },
                },
                "stop_reason": "final_verification_budget_unavailable",
            }
        ),
        encoding="utf-8",
    )

    result = AgentResult(
        task_id="track_a_residual_stream_deadlock",
        success=True,
        status="terminated_budget",
        termination_reason="final_verification_budget_unavailable",
        output_dir=str(tmp_path),
        trajectory=[
            TrajectoryEvent(
                step=1,
                stage="select_best",
                status="passed",
                details={
                    "selected_design": selected,
                    "selected_design_fully_verified": True,
                    "selected_design_frequency_compliant": True,
                    "selected_design_resource_compliant": True,
                },
            )
        ],
    )
    report_path = tmp_path / "track_a_score_estimate.json"
    report = {
        "selected_design": {
            "estimated_frequency_mhz": 357.65,
            "official_latency_cycles": 68,
        },
        "public_score_estimate": 3.1,
        "maximum_score": 4.0,
    }

    enriched = _enrich_track_a_result(result, report_path, report)

    assert result.success is True
    assert result.status == "completed_budget"
    assert (
        result.termination_reason
        == "verified_result_selected_after_budget_exhaustion"
    )
    assert enriched["final_design_verified"] is True

    payload = json.loads(
        (tmp_path / "unified_agent_result.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "completed_budget"
    assert payload["final_design_verified"] is True
    assert payload["selection_mode"] == "research_pareto"
    assert payload["meets_submission_frequency"] is True

    budget = json.loads(
        (tmp_path / "budget_summary.json").read_text(encoding="utf-8")
    )
    assert (
        budget["stop_reason"]
        == "verified_result_selected_after_budget_exhaustion"
    )


def test_agent_result_exposes_selection_compliance() -> None:
    result = AgentResult(
        task_id="task",
        success=True,
        status="success",
        termination_reason="done",
        output_dir="output",
        trajectory=[
            TrajectoryEvent(
                step=1,
                stage="select_best",
                status="passed",
                details={
                    "selected_design": {"candidate_index": 2},
                    "selected_design_fully_verified": True,
                    "selected_design_frequency_compliant": True,
                    "selected_design_resource_compliant": False,
                },
            )
        ],
    )

    payload = result.to_dict()

    assert payload["final_design_verified"] is True
    assert payload["selected_design_frequency_compliant"] is True
    assert payload["selected_design_resource_compliant"] is False
