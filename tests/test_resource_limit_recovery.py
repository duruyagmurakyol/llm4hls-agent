from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.parent_selection import select_refinement_parent
from agent.optimise.resource_recovery import (
    prepare_resource_recovery_prompt,
    resource_limit_recovery_trigger,
)
from agent.optimise.runner import _prepare_next_prompt


def _feasible_parent(index: int = 2) -> dict[str, object]:
    return {
        "candidate_index": index,
        "candidate_file": f"out/candidate_{index:03d}.cpp",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": True,
        "fully_verified": True,
        "meets_frequency_requirement": True,
        "resource_limit_compliance": {"passed": True, "violations": []},
        "verdict": "keep_pareto_candidate",
        "metrics": {
            "frequency_mhz": 134.318,
            "latency_ns": 56961.695,
            "throughput_period_ns": 57058.464,
            "resources_lut_used": 5419,
            "resources_ff_used": 8384,
            "resources_dsp_used": 62,
            "resources_bram_used": 0,
        },
    }


def _resource_rejection(index: int = 3) -> dict[str, object]:
    return {
        "candidate_index": index,
        "candidate_file": f"out/candidate_{index:03d}.cpp",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
        "cosim": None,
        "fully_verified": False,
        "meets_frequency_requirement": True,
        "meets_resource_limits": False,
        "verdict": "reject_resource_limits",
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
        "metrics": {
            "frequency_mhz": 127.599,
            "latency_ns": 8832.299,
            "throughput_period_ns": 8840.136,
            "resources_lut_used": 151392,
            "resources_ff_used": 277989,
            "resources_dsp_used": 453,
            "resources_bram_used": 0,
        },
    }


def test_resource_rejection_returns_to_feasible_pareto_parent() -> None:
    baseline = {
        **_feasible_parent(1),
        "verdict": "reject_no_change",
        "metrics": {
            **_feasible_parent(1)["metrics"],
            "latency_ns": 112819.882,
        },
    }
    parent = _feasible_parent()
    rejected = _resource_rejection()

    selected = select_refinement_parent([baseline, parent, rejected])

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 2
    assert reason == "resource_limit_recovery_from_feasible_pareto"


def test_duplicate_of_resource_rejection_keeps_recovery_active() -> None:
    parent = _feasible_parent()
    rejected = _resource_rejection()
    duplicate = {
        "candidate_index": 4,
        "verdict": "reject_duplicate",
        "duplicate_of": 3,
    }
    records = [parent, rejected, duplicate]

    assert resource_limit_recovery_trigger(records) == rejected
    selected = select_refinement_parent(records)

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 2
    assert reason == "resource_limit_recovery_from_feasible_pareto"


def test_resource_rejected_candidate_is_not_a_normal_fallback_parent() -> None:
    selected = select_refinement_parent(
        [
            {
                "candidate_index": 1,
                "static_validation": True,
                "csim": True,
                "fully_verified": False,
                "verdict": "reject_csim",
            },
            _resource_rejection(),
        ]
    )

    assert selected is not None
    record, reason = selected
    assert record["candidate_index"] == 1
    assert reason == "csim_passed_candidate"


def test_resource_recovery_prompt_contains_exact_violations(
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
                    "interpretation": "Aggressive parallelism replicated operators.",
                }
            }
        ),
        encoding="utf-8",
    )
    parent = _feasible_parent()
    rejected = _resource_rejection()
    (output / "experiment_summary.json").write_text(
        json.dumps(
            {
                "baseline_metrics": {},
                "candidates": [parent, rejected],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "benchmark": "gemm",
                "top_function": "kernel_gemm",
                "baseline": {"source": "baseline.cpp"},
                "output_dir": "out",
                "prompt_constraints": ["Preserve all 30 accumulation terms."],
            }
        ),
        encoding="utf-8",
    )

    prompt_path = prepare_resource_recovery_prompt(config, 2, 3, 4)
    prompt = prompt_path.read_text(encoding="utf-8")
    feedback = json.loads(
        (output / "candidate_004_resource_recovery_feedback.json").read_text(
            encoding="utf-8"
        )
    )
    strategy = json.loads(
        (output / "candidate_004_strategy.json").read_text(encoding="utf-8")
    )

    assert "Continue from candidate 002" in prompt
    assert "resources_lut_used: actual 151392.0, limit 70560.0, excess 80832.0" in prompt
    assert "resources_ff_used: actual 277989.0, limit 141120.0, excess 136869.0" in prompt
    assert "resources_dsp_used: actual 453.0, limit 360.0, excess 93.0" in prompt
    assert "smaller bounded UNROLL factor" in prompt
    assert "PIPELINE placement when it implies complete unrolling" in prompt
    assert "Do not reproduce the rejected candidate" in prompt
    assert "Preserve all 30 accumulation terms" in prompt
    assert feedback["parent_candidate_index"] == 2
    assert feedback["rejected_candidate_index"] == 3
    assert strategy["name"] == "recover_resource_limits"
    assert strategy["source_candidate_index"] == 2


def test_runner_dispatches_resource_recovery_prompt(monkeypatch, tmp_path: Path) -> None:
    called: dict[str, int] = {}

    def fake_prepare(config_source, parent_index, rejected_index, next_index):
        del config_source
        called.update(
            parent_index=parent_index,
            rejected_index=rejected_index,
            next_index=next_index,
        )
        return tmp_path / "prompt.txt"

    monkeypatch.setattr(
        "agent.optimise.runner.prepare_resource_recovery_prompt",
        fake_prepare,
    )

    parent = _feasible_parent()
    rejected = _resource_rejection()
    _prepare_next_prompt(
        tmp_path / "config.json",
        parent,
        2,
        4,
        {"candidates": [parent, rejected]},
        "resource_limit_recovery_from_feasible_pareto",
    )

    assert called == {
        "parent_index": 2,
        "rejected_index": 3,
        "next_index": 4,
    }
