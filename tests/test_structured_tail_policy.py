from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.optimise import structured_tail


def _record(
    index: int,
    *,
    refinement_eligible: bool,
    latency_ns: float,
    fully_verified: bool = True,
    resource_ok: bool = True,
    verdict: str = "keep_pareto_candidate",
) -> dict:
    return {
        "candidate_index": index,
        "candidate_file": f"out/candidate_{index:03d}.cpp",
        "refinement_eligible": refinement_eligible,
        "fully_verified": fully_verified,
        "verdict": verdict,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {
            "configured": True,
            "passed": resource_ok,
            "limits": {"resources_lut_used": 1000},
        },
        "metrics": {
            "latency_ns": latency_ns,
            "throughput_period_ns": latency_ns,
            "resources_lut_used": 500,
            "resources_ff_used": 500,
            "resources_dsp_used": 1,
            "resources_bram_used": 0,
        },
        "cost": {"total_tokens": 100, "tool_calls": 2, "tool_seconds": 1.0},
    }


def _config(tmp_path: Path) -> Path:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "candidate_001_prompt.txt").write_text(
        "BASELINE DIAGNOSIS\nComplete baseline source:\n"
        "int kernel(int x) { return x; }\n",
        encoding="utf-8",
    )
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "benchmark": "test",
                "top_function": "kernel",
                "output_dir": "run",
                "baseline": {"source": "baseline.cpp"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_c4_selects_best_explicitly_refinement_eligible_exploration() -> None:
    selected = structured_tail.select_structured_exploitation_parent(
        [
            _record(1, refinement_eligible=True, latency_ns=80.0),
            _record(2, refinement_eligible=True, latency_ns=60.0),
            _record(3, refinement_eligible=False, latency_ns=10.0),
            _record(4, refinement_eligible=True, latency_ns=1.0),
        ]
    )

    assert selected is not None
    parent, reason = selected
    assert parent["candidate_index"] == 2
    assert reason == structured_tail.STRUCTURED_EXPLOIT_REASON


def test_c4_does_not_promote_archive_only_candidate() -> None:
    selected = structured_tail.select_structured_exploitation_parent(
        [
            _record(1, refinement_eligible=False, latency_ns=10.0),
            _record(2, refinement_eligible=False, latency_ns=20.0),
            _record(3, refinement_eligible=False, latency_ns=30.0),
        ]
    )
    assert selected is None


def test_c5_accepts_only_bounded_recovery_reason() -> None:
    records = [_record(1, refinement_eligible=True, latency_ns=50.0)]

    def recovery_selector(values, selection):
        del values, selection
        return records[0], "pending_latency_recovery_strategy"

    selected = structured_tail.select_structured_recovery_parent(
        records,
        None,
        recovery_selector,
    )
    assert selected == (records[0], "pending_latency_recovery_strategy")


def test_c5_rejects_normal_refinement_reason() -> None:
    records = [_record(1, refinement_eligible=True, latency_ns=50.0)]

    def normal_selector(values, selection):
        del values, selection
        return records[0], "pareto_candidate"

    assert (
        structured_tail.select_structured_recovery_parent(
            records,
            None,
            normal_selector,
        )
        is None
    )


@pytest.mark.parametrize(
    ("candidate_index", "expected_reason", "expected_family"),
    [
        (
            4,
            structured_tail.STRUCTURED_EXPLOIT_FALLBACK_REASON,
            "loop_schedule_restructuring",
        ),
        (
            5,
            structured_tail.STRUCTURED_RECOVERY_FALLBACK_REASON,
            "pipeline_dataflow_restructuring",
        ),
    ],
)
def test_baseline_fallback_parent_is_slot_specific(
    candidate_index: int,
    expected_reason: str,
    expected_family: str,
) -> None:
    parent, reason = structured_tail.baseline_fallback_parent(
        {"baseline": {"source": "baseline.cpp"}},
        candidate_index=candidate_index,
    )
    assert parent["candidate_index"] == 0
    assert parent["candidate_file"] == "baseline.cpp"
    assert parent["strategy_family"] == expected_family
    assert reason == expected_reason


def test_c5_baseline_fallback_prompt_is_independent_and_audited(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(structured_tail, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)

    for index, name in enumerate(
        [
            "critical_path_restructuring",
            "bounded_unroll",
            "memory_parallelism",
            "focused_exploitation",
        ],
        1,
    ):
        (tmp_path / "run" / f"candidate_{index:03d}_strategy.json").write_text(
            json.dumps({"name": name}),
            encoding="utf-8",
        )

    prompt_path = structured_tail.prepare_structured_baseline_fallback_prompt(
        config,
        candidate_index=5,
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    assert "Implementation parent: original verified baseline (candidate 000)." in prompt
    assert "Strategy family: pipeline_dataflow_restructuring." in prompt
    assert "critical_path_restructuring" in prompt
    assert "bounded_unroll" in prompt
    assert "memory_parallelism" in prompt

    strategy = json.loads(
        (tmp_path / "run" / "candidate_005_strategy.json").read_text(
            encoding="utf-8"
        )
    )
    assert strategy["source_candidate_index"] == 0
    assert strategy["compliance_mode"] == "advisory"
    assert strategy["phase"] == "recovery_fallback"

    feedback = json.loads(
        (tmp_path / "run" / "candidate_005_feedback.json").read_text(
            encoding="utf-8"
        )
    )
    assert feedback["selected_parent"] == "verified_baseline"
    assert feedback["structured_schedule"] is True


def test_search_decision_records_parent_and_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(structured_tail, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)
    parent = _record(2, refinement_eligible=True, latency_ns=50.0)

    path = structured_tail.write_structured_search_decision(
        config,
        candidate_index=4,
        phase="exploit",
        parent=parent,
        reason=structured_tail.STRUCTURED_EXPLOIT_REASON,
    )
    decision = json.loads(path.read_text(encoding="utf-8"))
    assert decision["candidate_index"] == 4
    assert decision["phase"] == "exploit"
    assert decision["selected_parent_index"] == 2
    assert decision["parent_reason"] == structured_tail.STRUCTURED_EXPLOIT_REASON
