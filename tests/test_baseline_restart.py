from __future__ import annotations

import json
from pathlib import Path

from agent.optimise import diagnose
from agent.optimise.parent_selection import (
    BASELINE_RESTART_REASON,
    select_refinement_parent,
)


def record(
    index: int,
    verdict: str,
    *,
    fully_verified: bool = True,
    static_validation: bool | None = True,
) -> dict:
    return {
        "candidate_index": index,
        "candidate_file": f"candidate_{index:03d}.cpp",
        "verdict": verdict,
        "fully_verified": fully_verified,
        "static_validation": static_validation,
        "csim": fully_verified,
        "synthesis": fully_verified,
        "refinement_eligible": False,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {
            "configured": False,
            "passed": True,
            "limits": {},
            "usage": {},
            "violations": [],
        },
        "metrics": {
            "latency_ns": 100.0 + index,
            "throughput_period_ns": 100.0 + index,
            "resources_lut_used": 100 + index,
            "resources_ff_used": 50 + index,
            "resources_dsp_used": 1,
            "resources_bram_used": 0,
        },
        "cost": {
            "total_tokens": 100,
            "tool_calls": 2,
            "tool_seconds": 1.0,
        },
    }


def _assert_baseline_restart(verdict: str) -> None:
    selected = select_refinement_parent([record(1, verdict)])

    assert selected is not None
    parent, reason = selected

    assert parent["candidate_index"] == 0
    assert parent["trigger_candidate_index"] == 1
    assert parent["trigger_verdict"] == verdict
    assert reason == BASELINE_RESTART_REASON


def test_no_objective_gain_restarts_from_baseline() -> None:
    _assert_baseline_restart("reject_no_objective_gain")


def test_no_change_restarts_from_baseline() -> None:
    _assert_baseline_restart("reject_no_change")


def test_duplicate_restarts_from_baseline() -> None:
    _assert_baseline_restart("reject_duplicate")


def test_latest_no_gain_does_not_override_pareto_parent() -> None:
    selected = select_refinement_parent(
        [
            record(1, "keep_pareto_candidate"),
            record(2, "reject_no_objective_gain"),
        ]
    )

    assert selected is not None
    parent, reason = selected

    assert parent["candidate_index"] == 1
    assert reason == "pareto_candidate"


def test_latest_duplicate_does_not_override_pareto_parent() -> None:
    selected = select_refinement_parent(
        [
            record(1, "keep_pareto_candidate"),
            record(2, "reject_duplicate", fully_verified=False),
        ]
    )

    assert selected is not None
    parent, reason = selected

    assert parent["candidate_index"] == 1
    assert reason == "pareto_candidate"


def test_latest_static_failure_remains_repairable() -> None:
    selected = select_refinement_parent(
        [
            record(1, "reject_no_objective_gain"),
            record(
                2,
                "reject_static",
                fully_verified=False,
                static_validation=True,
            ),
        ]
    )

    assert selected is not None
    parent, _ = selected

    assert parent["candidate_index"] == 2


def test_restart_prompt_uses_baseline_not_rejected_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(diagnose, "REPO_ROOT", tmp_path)

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    baseline = tmp_path / "baseline.cpp"
    baseline.write_text(
        "int kernel(int x) { return x; }\n",
        encoding="utf-8",
    )

    rejected = output_dir / "candidate_001.cpp"
    rejected.write_text(
        "// BAD_PARENT_MARKER\n"
        "#pragma HLS unroll\n"
        "int kernel(int x) { return x; }\n",
        encoding="utf-8",
    )

    baseline_metrics = {
        "clock_period_ns": 4.0,
        "latency_best_cycles": 100,
        "latency_average_cycles": 100,
        "latency_worst_cycles": 100,
        "interval_min_cycles": 100,
        "interval_max_cycles": 100,
        "resources_lut_used": 100,
        "resources_ff_used": 50,
        "resources_dsp_used": 1,
        "resources_bram_used": 0,
    }
    rejected_metrics = {
        **baseline_metrics,
        "clock_period_ns": 7.0,
        "latency_best_cycles": 200,
        "resources_lut_used": 1000,
        "resources_ff_used": 800,
    }

    (output_dir / "baseline_source_target.json").write_text(
        json.dumps(
            {
                "target_name": "kernel",
                "loop_label": None,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "baseline_source_cause.json").write_text(
        json.dumps(
            {
                "primary_hypothesis": {
                    "category": "dependency",
                    "interpretation": (
                        "A loop dependency may limit throughput."
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(
            {
                "baseline_metrics": baseline_metrics,
                "candidates": [
                    {
                        "candidate_index": 1,
                        "candidate_file": "out/candidate_001.cpp",
                        "verdict": "reject_no_objective_gain",
                        "fully_verified": True,
                        "metrics": rejected_metrics,
                        "deltas_percent": {
                            "latency_ns": 100.0,
                            "throughput_period_ns": 100.0,
                            "resources_lut_used": 900.0,
                            "resources_ff_used": 1500.0,
                            "resources_dsp_used": 0.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "benchmark": "test",
                "top_function": "kernel",
                "output_dir": "out",
                "baseline": {
                    "source": "baseline.cpp",
                },
            }
        ),
        encoding="utf-8",
    )

    prompt_path = diagnose.prepare_refinement_prompt(config, 0, 2)
    prompt = prompt_path.read_text(encoding="utf-8")
    feedback = json.loads(
        (output_dir / "candidate_002_feedback.json").read_text(
            encoding="utf-8"
        )
    )

    assert (
        "Implementation parent: original verified baseline"
        in prompt
    )
    assert "int kernel(int x) { return x; }" in prompt
    assert "BAD_PARENT_MARKER" not in prompt
    assert "Measured evidence from rejected candidate 001" in prompt

    assert feedback["previous_candidate_index"] == 0
    assert feedback["evidence_candidate_index"] == 1
    assert feedback["restart_from_baseline"] is True


def test_synthesis_equivalent_restarts_from_baseline() -> None:
    _assert_baseline_restart("reject_synthesis_equivalent")
