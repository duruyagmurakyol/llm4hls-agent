from pathlib import Path

from agent.config import TaskManifest
from agent.controller import _record_phase_transitions
from agent.state import AgentPhase, AgentResult, TrajectoryEvent


def _task(tmp_path: Path, adapter_kind: str) -> TaskManifest:
    return TaskManifest(
        path=tmp_path / "task.json",
        data={
            "task_id": "phase_test",
            "task_kind": "functional_failure",
            "adapter": {"kind": adapter_kind},
            "output_dir": str(tmp_path / "output"),
        },
    )


def _result(trajectory: list[TrajectoryEvent]) -> AgentResult:
    return AgentResult(
        task_id="phase_test",
        success=True,
        status="fully_verified",
        termination_reason="completed",
        output_dir="output",
        trajectory=trajectory,
    )


def test_direct_repair_records_explicit_phases(tmp_path: Path) -> None:
    result = _record_phase_transitions(
        _task(tmp_path, "direct_api_repair"),
        _result(
            [
                TrajectoryEvent(1, "repair", "passed", {"failure_class": "functional"}),
                TrajectoryEvent(
                    2,
                    "post_repair_synthesis",
                    "passed",
                    {"candidate_hash": "a" * 64, "return_code": 0},
                ),
                TrajectoryEvent(
                    3,
                    "post_repair_cosim",
                    "passed",
                    {"candidate_hash": "a" * 64, "return_code": 0},
                ),
            ]
        ),
    )

    assert result.current_phase is AgentPhase.TERMINATE
    assert [transition.to_phase for transition in result.phase_transitions] == [
        AgentPhase.DISCOVER,
        AgentPhase.DIAGNOSE,
        AgentPhase.REPAIR,
        AgentPhase.ESTABLISH_BASELINE,
        AgentPhase.TERMINATE,
    ]
    assert result.phase_transitions[3].details["candidate_hash"] == "a" * 64
    assert result.to_dict()["current_phase"] == "terminate"
    assert result.to_dict()["phase_transitions"][0]["from_phase"] is None


def test_auto_repair_records_initial_validation_and_diagnosis(tmp_path: Path) -> None:
    result = _record_phase_transitions(
        _task(tmp_path, "auto"),
        _result(
            [
                TrajectoryEvent(1, "initial_csim", "failed", {"failure_class": "functional"}),
                TrajectoryEvent(2, "repair", "passed", {"failure_class": "functional"}),
                TrajectoryEvent(3, "post_repair_synthesis", "passed", {}),
            ]
        ),
    )

    assert [transition.to_phase for transition in result.phase_transitions] == [
        AgentPhase.DISCOVER,
        AgentPhase.VALIDATE_INITIAL,
        AgentPhase.DIAGNOSE,
        AgentPhase.REPAIR,
        AgentPhase.ESTABLISH_BASELINE,
        AgentPhase.TERMINATE,
    ]


def test_ppa_records_generation_validation_and_selection(tmp_path: Path) -> None:
    result = _record_phase_transitions(
        _task(tmp_path, "autonomous_ppa"),
        _result(
            [
                TrajectoryEvent(1, "generation", "passed", {"candidate": 1}),
                TrajectoryEvent(2, "static_validation", "passed", {"candidate": 1}),
                TrajectoryEvent(3, "synthesis", "passed", {"candidate": 1}),
            ]
        ),
    )

    assert [transition.to_phase for transition in result.phase_transitions] == [
        AgentPhase.DISCOVER,
        AgentPhase.ESTABLISH_BASELINE,
        AgentPhase.DIAGNOSE_PPA,
        AgentPhase.GENERATE_OPTIMISATION,
        AgentPhase.VALIDATE_CANDIDATE,
        AgentPhase.SELECT_BEST,
        AgentPhase.TERMINATE,
    ]
