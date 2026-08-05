from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.parent_selection import select_refinement_parent
from agent.optimise.refinement_strategy import check_strategy_compliance
from agent.optimise.resource_recovery import (
    RESOURCE_FREQUENCY_BALANCE_REASON,
    prepare_resource_frequency_balance_prompt,
    resource_frequency_balance_trigger,
)
from agent.optimise.runner import _prepare_next_prompt


def _feasible_parent() -> dict[str, object]:
    return {
        "candidate_index": 2,
        "candidate_file": "out/candidate_002.cpp",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": True,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {"passed": True, "violations": []},
        "verdict": "keep_pareto_candidate",
        "metrics": {
            "frequency_mhz": 134.31833445265278,
            "clock_period_ns": 7.445,
            "latency_ns": 56961.695,
            "throughput_period_ns": 57058.464,
            "resources_lut_used": 5419,
            "resources_ff_used": 8384,
            "resources_dsp_used": 62,
            "resources_bram_used": 0,
        },
    }


def _resource_rejection() -> dict[str, object]:
    return {
        "candidate_index": 3,
        "candidate_file": "out/candidate_003.cpp",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "fully_verified": False,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {
            "passed": False,
            "violations": [
                {
                    "metric": "resources_lut_used",
                    "actual": 151392.0,
                    "limit": 70560.0,
                    "excess": 80832.0,
                    "reason": "limit_exceeded",
                },
                {
                    "metric": "resources_ff_used",
                    "actual": 277989.0,
                    "limit": 141120.0,
                    "excess": 136869.0,
                    "reason": "limit_exceeded",
                },
                {
                    "metric": "resources_dsp_used",
                    "actual": 453.0,
                    "limit": 360.0,
                    "excess": 93.0,
                    "reason": "limit_exceeded",
                },
            ],
        },
        "verdict": "reject_resource_limits",
        "metrics": {
            "frequency_mhz": 127.59984688018375,
            "clock_period_ns": 7.837,
            "latency_ns": 8832.299,
            "throughput_period_ns": 8840.136,
            "resources_lut_used": 151392,
            "resources_ff_used": 277989,
            "resources_dsp_used": 453,
            "resources_bram_used": 0,
        },
    }


def _frequency_rejection() -> dict[str, object]:
    return {
        "candidate_index": 6,
        "candidate_file": "out/candidate_006.cpp",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "fully_verified": False,
        "meets_frequency_requirement": False,
        "resource_limit_compliance": {"passed": True, "violations": []},
        "verdict": "reject_frequency_threshold",
        "metrics": {
            "frequency_mhz": 12.709387153351464,
            "clock_period_ns": 78.682,
            "latency_ns": 1181095.502,
            "throughput_period_ns": 1181016.82,
            "resources_lut_used": 1776,
            "resources_ff_used": 1625,
            "resources_dsp_used": 38,
            "resources_bram_used": 0,
        },
    }


def _write_source_strategy(output: Path) -> None:
    (output / "candidate_006_strategy.json").write_text(
        json.dumps(
            {
                "name": "recover_resource_limits",
                "parameters": {
                    "rejected_candidate_index": 3,
                    "violations": (
                        _resource_rejection()["resource_limit_compliance"][
                            "violations"
                        ]
                    ),
                },
                "source_candidate_index": 2,
                "next_candidate_index": 6,
                "trigger": "resource_limit_violation",
            }
        ),
        encoding="utf-8",
    )


def test_failed_resource_recovery_returns_to_original_feasible_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _write_source_strategy(output)
    monkeypatch.setattr("agent.optimise.resource_recovery.REPO_ROOT", tmp_path)
    monkeypatch.setattr("agent.optimise.parent_selection.REPO_ROOT", tmp_path)

    records = [_feasible_parent(), _resource_rejection(), _frequency_rejection()]
    trigger = resource_frequency_balance_trigger(records)
    selected = select_refinement_parent(records)

    assert trigger is not None
    assert trigger["parent"]["candidate_index"] == 2
    assert trigger["resource_rejected"]["candidate_index"] == 3
    assert trigger["frequency_rejected"]["candidate_index"] == 6
    assert selected is not None
    parent, reason = selected
    assert parent["candidate_index"] == 2
    assert reason == RESOURCE_FREQUENCY_BALANCE_REASON


def test_duplicate_of_failed_recovery_preserves_balanced_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _write_source_strategy(output)
    monkeypatch.setattr("agent.optimise.resource_recovery.REPO_ROOT", tmp_path)
    monkeypatch.setattr("agent.optimise.parent_selection.REPO_ROOT", tmp_path)

    duplicate = {
        "candidate_index": 7,
        "candidate_file": "out/candidate_007.cpp",
        "verdict": "reject_duplicate",
        "duplicate_of": 6,
    }
    selected = select_refinement_parent(
        [_feasible_parent(), _resource_rejection(), _frequency_rejection(), duplicate]
    )

    assert selected is not None
    parent, reason = selected
    assert parent["candidate_index"] == 2
    assert reason == RESOURCE_FREQUENCY_BALANCE_REASON


def test_balanced_prompt_contains_both_boundaries_and_hard_constraints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.resource_recovery.REPO_ROOT", tmp_path)
    output = tmp_path / "out"
    output.mkdir()

    source = (
        '#include "gemm.h"\n'
        "void kernel_gemm(double alpha, double beta, double C[20][25], "
        "double A[20][30], double B[30][25]) { C[0][0] += alpha + beta; }\n"
    )
    (tmp_path / "baseline.cpp").write_text(source, encoding="utf-8")
    (output / "candidate_002.cpp").write_text(
        source.replace("alpha + beta", "beta + alpha"),
        encoding="utf-8",
    )
    (output / "baseline_source_target.json").write_text(
        json.dumps({"target_name": "kernel_gemm", "loop_label": None}),
        encoding="utf-8",
    )
    (output / "baseline_source_cause.json").write_text(
        json.dumps(
            {
                "primary_hypothesis": {
                    "category": "loop_parallelism",
                    "interpretation": "The loop nest needs bounded local parallelism.",
                }
            }
        ),
        encoding="utf-8",
    )
    records = [_feasible_parent(), _resource_rejection(), _frequency_rejection()]
    (output / "experiment_summary.json").write_text(
        json.dumps({"baseline_metrics": {}, "candidates": records}),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "benchmark": "gemm",
                "top_function": "kernel_gemm",
                "minimum_frequency_mhz": 100.0,
                "target_clock_period_ns": 10.0,
                "resource_limits": {
                    "lut": 70560,
                    "ff": 141120,
                    "dsp": 360,
                    "bram_18k": 432,
                },
                "baseline": {"source": "baseline.cpp"},
                "output_dir": "out",
                "prompt_constraints": ["Preserve all 30 accumulation terms."],
            }
        ),
        encoding="utf-8",
    )

    prompt_path = prepare_resource_frequency_balance_prompt(
        config,
        2,
        3,
        6,
        7,
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    feedback = json.loads(
        (
            output
            / "candidate_007_resource_frequency_balance_feedback.json"
        ).read_text(encoding="utf-8")
    )
    strategy = json.loads(
        (output / "candidate_007_strategy.json").read_text(encoding="utf-8")
    )

    assert "Use candidate 002 as the only source architecture" in prompt
    assert "Candidate 003 was over-parallel" in prompt
    assert "Candidate 006 reduced resources but failed timing" in prompt
    assert "excess 80832.0" in prompt
    assert "Frequency shortfall: 87.29061284664854 MHz" in prompt
    assert "frequency >= 100.0 MHz" in prompt
    assert "clock period <= 10.0 ns" in prompt
    assert "Seek latency below the feasible parent's 56961.695 ns" in prompt
    assert "Do not force II=1" in prompt
    assert "moderate partial unrolling" in prompt
    assert "Preserve all 30 accumulation terms" in prompt
    assert feedback["parent_candidate_index"] == 2
    assert feedback["resource_rejected_candidate_index"] == 3
    assert feedback["frequency_rejected_candidate_index"] == 6
    assert strategy["name"] == "recover_resource_frequency_balance"
    assert strategy["source_candidate_index"] == 2


def test_runner_dispatches_balanced_recovery_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    called: dict[str, int] = {}
    parent = _feasible_parent()
    resource_rejected = _resource_rejection()
    frequency_rejected = _frequency_rejection()

    monkeypatch.setattr(
        "agent.optimise.runner.resource_frequency_balance_trigger",
        lambda records: {
            "parent": parent,
            "resource_rejected": resource_rejected,
            "frequency_rejected": frequency_rejected,
            "source_strategy": {},
        },
    )

    def fake_prepare(
        config_source,
        parent_index,
        resource_rejected_index,
        frequency_rejected_index,
        next_index,
    ):
        del config_source
        called.update(
            parent_index=parent_index,
            resource_rejected_index=resource_rejected_index,
            frequency_rejected_index=frequency_rejected_index,
            next_index=next_index,
        )
        return tmp_path / "prompt.txt"

    monkeypatch.setattr(
        "agent.optimise.runner.prepare_resource_frequency_balance_prompt",
        fake_prepare,
    )

    _prepare_next_prompt(
        tmp_path / "config.json",
        parent,
        2,
        7,
        {"candidates": [parent, resource_rejected, frequency_rejected]},
        RESOURCE_FREQUENCY_BALANCE_REASON,
    )

    assert called == {
        "parent_index": 2,
        "resource_rejected_index": 3,
        "frequency_rejected_index": 6,
        "next_index": 7,
    }


def test_balanced_strategy_is_prompt_driven_not_static_rejected() -> None:
    result = check_strategy_compliance(
        "void kernel() {}\n",
        {
            "name": "recover_resource_frequency_balance",
            "parameters": {},
        },
    )

    assert result["required"] is False
    assert result["passed"] is True
    assert result["reason"] == "requires_post_synthesis_evidence"
