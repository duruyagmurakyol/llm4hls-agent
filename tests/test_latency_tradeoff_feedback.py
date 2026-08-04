from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.diagnose import (
    _recover_latency_tradeoff_strategy,
    prepare_tradeoff_prompt,
)
from agent.optimise.generate import _attach_latency_recovery_factor
from agent.optimise.refinement_strategy import (
    apply_strategy_directives,
    check_strategy_compliance,
    select_latency_recovery_factor,
)


def _pareto_record(
    *,
    latency_delta: float,
    dsp_delta: float,
) -> dict:
    return {
        "candidate_index": 1,
        "verdict": "keep_pareto_candidate",
        "fully_verified": True,
        "metrics": {
            "clock_period_ns": 7.584,
            "latency_best_cycles": 2623,
            "latency_average_cycles": 2623,
            "latency_worst_cycles": 2623,
            "latency_ns": 19892.832,
            "interval_min_cycles": 2624,
            "interval_max_cycles": 2624,
            "throughput_period_ns": 19900.416,
            "resources_lut_used": 5924,
            "resources_ff_used": 12484,
            "resources_dsp_used": 28,
            "resources_bram_used": 0,
        },
        "deltas_percent": {
            "latency_ns": latency_delta,
            "throughput_period_ns": latency_delta,
            "resources_lut_used": -32.23,
            "resources_ff_used": -42.54,
            "resources_dsp_used": dsp_delta,
            "resources_bram_used": 0.0,
        },
    }


def test_slow_resource_saving_pareto_candidate_generates_latency_recovery_feedback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.diagnose.REPO_ROOT", tmp_path)
    output = tmp_path / "out"
    output.mkdir()

    source = (
        '#include "bicg.h"\n'
        "void kernel_bicg(double a[4][4], double s[4], double q[4], "
        "double p[4], double r[4]) {\n"
        "  for (int i = 0; i < 4; ++i) q[i] = r[i];\n"
        "}\n"
    )
    (tmp_path / "baseline.cpp").write_text(source, encoding="utf-8")
    (output / "candidate_001.cpp").write_text(
        source.replace("q[i] = r[i]", "q[i] = r[i] + 0.0"),
        encoding="utf-8",
    )
    (output / "candidate_001_synthesis.json").write_text(
        json.dumps({"passed": True, "metrics": _pareto_record(
            latency_delta=148.2,
            dsp_delta=-50.0,
        )["metrics"]}),
        encoding="utf-8",
    )
    record = _pareto_record(latency_delta=148.2, dsp_delta=-50.0)
    (output / "experiment_summary.json").write_text(
        json.dumps(
            {
                "baseline_metrics": {
                    "clock_period_ns": 7.584,
                    "latency_best_cycles": 1057,
                    "latency_average_cycles": 1057,
                    "latency_worst_cycles": 1057,
                    "latency_ns": 8016.288,
                    "interval_min_cycles": 1058,
                    "interval_max_cycles": 1058,
                    "throughput_period_ns": 8023.872,
                    "resources_lut_used": 8741,
                    "resources_ff_used": 21727,
                    "resources_dsp_used": 56,
                    "resources_bram_used": 0,
                },
                "candidates": [record],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "benchmark": "bicg",
                "top_function": "kernel_bicg",
                "baseline": {"source": "baseline.cpp"},
                "output_dir": "out",
            }
        ),
        encoding="utf-8",
    )

    prompt_path = prepare_tradeoff_prompt(config, 1, 2)
    prompt = prompt_path.read_text(encoding="utf-8")
    strategy = json.loads(
        (output / "candidate_002_strategy.json").read_text(encoding="utf-8")
    )
    feedback = json.loads(
        (output / "candidate_002_tradeoff_feedback.json").read_text(
            encoding="utf-8"
        )
    )

    assert "Selected strategy: recover_latency_tradeoff" in prompt
    assert "Restore controlled parallelism" in prompt
    assert "resources_lut_used: -32.230%" in prompt
    assert "resources_ff_used: -42.540%" in prompt
    assert "resources_dsp_used: -50.000%" in prompt
    assert "Do not revert completely to the baseline architecture" in prompt
    assert strategy["name"] == "recover_latency_tradeoff"
    assert strategy["source_candidate_index"] == 1
    assert strategy["next_candidate_index"] == 2
    assert strategy["trigger"] == "slow_resource_saving_pareto_candidate"
    assert strategy["preserve"] == [
        "resources_lut_used",
        "resources_ff_used",
        "resources_dsp_used",
    ]
    assert strategy["improve"] == ["latency_ns", "throughput_period_ns"]
    assert feedback["source_candidate_index"] == 1
    assert feedback["next_candidate_index"] == 2
    assert feedback["selected_strategy"]["name"] == "recover_latency_tradeoff"


def test_latency_recovery_trigger_requires_both_thresholds() -> None:
    assert _recover_latency_tradeoff_strategy(
        _pareto_record(latency_delta=50.0, dsp_delta=-25.0)
    ) is not None
    assert _recover_latency_tradeoff_strategy(
        _pareto_record(latency_delta=49.9, dsp_delta=-50.0)
    ) is None

    no_large_resource_saving = _pareto_record(
        latency_delta=80.0,
        dsp_delta=-24.9,
    )
    no_large_resource_saving["deltas_percent"].update(
        resources_lut_used=-10.0,
        resources_ff_used=-20.0,
    )
    assert _recover_latency_tradeoff_strategy(no_large_resource_saving) is None


def test_latency_recovery_factor_ladder_is_bounded() -> None:
    assert select_latency_recovery_factor([]) == 2
    assert select_latency_recovery_factor([2]) == 4
    assert select_latency_recovery_factor([2, 4]) == 8
    assert select_latency_recovery_factor([2, 4, 8]) is None


def test_generation_persists_next_untried_latency_recovery_factor(
    tmp_path: Path,
) -> None:
    def attach(index: int) -> dict:
        path = tmp_path / f"candidate_{index:03d}_strategy.json"
        strategy = {
            "name": "recover_latency_tradeoff",
            "parameters": {"latency_regression_percent": 148.2},
        }
        path.write_text(json.dumps(strategy), encoding="utf-8")
        return _attach_latency_recovery_factor(tmp_path, path, strategy)

    assert attach(2)["parameters"]["factor"] == 2
    assert attach(3)["parameters"]["factor"] == 4
    assert attach(4)["parameters"]["factor"] == 8
    exhausted = attach(5)
    assert "factor" not in exhausted["parameters"]

    persisted = json.loads(
        (tmp_path / "candidate_004_strategy.json").read_text(encoding="utf-8")
    )
    assert persisted["parameters"]["factor"] == 8


def test_latency_recovery_factor_is_applied_and_checked() -> None:
    source = (
        "void kernel(int a[8]) {\n"
        "  for (int i = 0; i < 8; ++i) {\n"
        "    a[i] += 1;\n"
        "  }\n"
        "}\n"
    )
    strategy = {
        "name": "recover_latency_tradeoff",
        "parameters": {"factor": 4},
    }

    rewritten = apply_strategy_directives(source, strategy)
    result = check_strategy_compliance(rewritten, strategy)

    assert "#pragma HLS PIPELINE II=1" in rewritten
    assert "#pragma HLS UNROLL factor=4" in rewritten
    assert result["required"] is True
    assert result["passed"] is True
    assert result["strategy"] == "recover_latency_tradeoff"
    assert result["expected"]["factor"] == 4


def test_latency_recovery_strategy_requires_post_synthesis_evidence() -> None:
    result = check_strategy_compliance(
        "void kernel() {}\n",
        {"name": "recover_latency_tradeoff", "parameters": {}},
    )

    assert result == {
        "required": False,
        "passed": True,
        "strategy": "recover_latency_tradeoff",
        "reason": "requires_post_synthesis_evidence",
    }
