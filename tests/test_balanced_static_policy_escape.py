from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.parent_selection import select_refinement_parent
from agent.optimise.resource_recovery import (
    RESOURCE_FREQUENCY_BALANCE_REASON,
    prepare_resource_frequency_balance_prompt,
    resource_frequency_balance_trigger,
)


def _parent() -> dict[str, object]:
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


def _resource_boundary() -> dict[str, object]:
    return {
        "candidate_index": 3,
        "candidate_file": "out/candidate_003.cpp",
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


def _frequency_boundary() -> dict[str, object]:
    return {
        "candidate_index": 6,
        "candidate_file": "out/candidate_006.cpp",
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


def _static_rejection() -> dict[str, object]:
    return {
        "candidate_index": 8,
        "candidate_file": "out/candidate_008.cpp",
        "static_validation": False,
        "fully_verified": False,
        "verdict": "reject_static",
        "reason": (
            "Static validation failed: "
            "no_complete_partition_on_interface_arrays"
        ),
    }


def _write_lineage(output: Path) -> None:
    (output / "candidate_008_strategy.json").write_text(
        json.dumps(
            {
                "name": "recover_resource_frequency_balance",
                "parameters": {
                    "resource_rejected_candidate_index": 3,
                    "frequency_rejected_candidate_index": 6,
                    "duplicate_retry_of_candidate_index": 7,
                    "duplicate_of_candidate_index": 2,
                },
                "source_candidate_index": 2,
                "next_candidate_index": 8,
                "trigger": "resource_frequency_balance_duplicate_escape",
            }
        ),
        encoding="utf-8",
    )
    (output / "candidate_008_static_validation.json").write_text(
        json.dumps(
            {
                "candidate_index": 8,
                "passed": False,
                "checks": {
                    "signature_preserved": True,
                    "no_complete_partition_on_interface_arrays": False,
                },
                "partition_guard_enabled": True,
                "complete_partition_issues": [
                    {
                        "variable": "B",
                        "pragma": (
                            "#pragma HLS ARRAY_PARTITION variable=B "
                            "complete dim=2"
                        ),
                        "reason": (
                            "complete partitioning of a top-level "
                            "interface array"
                        ),
                    },
                    {
                        "variable": "C",
                        "pragma": (
                            "#pragma HLS ARRAY_PARTITION variable=C "
                            "complete dim=2"
                        ),
                        "reason": (
                            "complete partitioning of a top-level "
                            "interface array"
                        ),
                    },
                    {
                        "variable": "A",
                        "pragma": (
                            "#pragma HLS ARRAY_PARTITION variable=A "
                            "complete dim=2"
                        ),
                        "reason": (
                            "complete partitioning of a top-level "
                            "interface array"
                        ),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_static_policy_failure_preserves_balanced_parent_and_boundaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _write_lineage(output)
    monkeypatch.setattr("agent.optimise.resource_recovery.REPO_ROOT", tmp_path)
    monkeypatch.setattr("agent.optimise.parent_selection.REPO_ROOT", tmp_path)

    records = [
        _parent(),
        _resource_boundary(),
        _frequency_boundary(),
        _static_rejection(),
    ]
    trigger = resource_frequency_balance_trigger(records)
    selected = select_refinement_parent(records)

    assert trigger is not None
    assert trigger["parent"]["candidate_index"] == 2
    assert trigger["resource_rejected"]["candidate_index"] == 3
    assert trigger["frequency_rejected"]["candidate_index"] == 6
    assert trigger["static_policy_escape"]["candidate_index"] == 8
    assert trigger["static_policy_escape"]["failed_checks"] == [
        "no_complete_partition_on_interface_arrays"
    ]
    assert len(
        trigger["static_policy_escape"]["complete_partition_issues"]
    ) == 3
    assert selected is not None
    parent, reason = selected
    assert parent["candidate_index"] == 2
    assert reason == RESOURCE_FREQUENCY_BALANCE_REASON


def test_next_prompt_forbids_interface_complete_partition_and_uses_local_buffers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _write_lineage(output)
    monkeypatch.setattr("agent.optimise.resource_recovery.REPO_ROOT", tmp_path)

    source = (
        '#include "gemm.h"\n'
        "void kernel_gemm(double alpha, double beta, double C[20][25], "
        "double A[20][30], double B[30][25]) { C[0][0] += alpha + beta; }\n"
    )
    (tmp_path / "baseline.cpp").write_text(source, encoding="utf-8")
    (output / "candidate_002.cpp").write_text(source, encoding="utf-8")
    (output / "baseline_source_target.json").write_text(
        json.dumps({"target_name": "kernel_gemm", "loop_label": None}),
        encoding="utf-8",
    )
    (output / "baseline_source_cause.json").write_text(
        json.dumps(
            {
                "primary_hypothesis": {
                    "category": "loop_parallelism",
                    "interpretation": "Use bounded local parallelism.",
                }
            }
        ),
        encoding="utf-8",
    )
    records = [
        _parent(),
        _resource_boundary(),
        _frequency_boundary(),
        _static_rejection(),
    ]
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
                "prompt_constraints": [
                    "Do not completely partition top-level interface arrays."
                ],
            }
        ),
        encoding="utf-8",
    )

    prompt_path = prepare_resource_frequency_balance_prompt(
        config,
        2,
        3,
        6,
        9,
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    feedback = json.loads(
        (
            output
            / "candidate_009_resource_frequency_balance_feedback.json"
        ).read_text(encoding="utf-8")
    )
    strategy = json.loads(
        (output / "candidate_009_strategy.json").read_text(encoding="utf-8")
    )

    assert "Candidate 008 was rejected before CSim and synthesis" in prompt
    assert "completely partitioned top-level interface arrays" in prompt
    assert "#pragma HLS ARRAY_PARTITION variable=A complete dim=2" in prompt
    assert "#pragma HLS ARRAY_PARTITION variable=B complete dim=2" in prompt
    assert "#pragma HLS ARRAY_PARTITION variable=C complete dim=2" in prompt
    assert "copy only a bounded row, column, or tile into a local array" in prompt
    assert "partition only that local array" in prompt
    assert "bounded block/cyclic factor" in prompt
    assert "Do not return candidate 002 unchanged" in prompt
    assert feedback["static_policy_escape"]["candidate_index"] == 8
    assert strategy["parameters"]["static_policy_retry_of_candidate_index"] == 8
    assert strategy["parameters"]["static_policy_failed_checks"] == [
        "no_complete_partition_on_interface_arrays"
    ]
    assert strategy["trigger"] == (
        "resource_frequency_balance_static_policy_escape"
    )
