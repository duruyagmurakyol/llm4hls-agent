from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.diagnose import prepare_refinement_prompt
from agent.optimise.refinement_strategy import check_strategy_compliance


def test_promising_frequency_violation_generates_recover_frequency_feedback(
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
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "clock_period_ns": 63.5,
                    "latency_best_cycles": 999,
                    "interval_min_cycles": 999,
                    "resources_lut_used": 2468,
                    "resources_ff_used": 1985,
                    "resources_dsp_used": 56,
                    "resources_bram_used": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (output / "baseline_source_target.json").write_text(
        json.dumps({"target_name": "kernel_bicg", "loop_label": None}),
        encoding="utf-8",
    )
    (output / "baseline_source_cause.json").write_text(
        json.dumps(
            {
                "primary_hypothesis": {
                    "category": "critical_path",
                    "interpretation": "The selected structure has a long combinational path.",
                }
            }
        ),
        encoding="utf-8",
    )
    (output / "experiment_summary.json").write_text(
        json.dumps(
            {
                "baseline_metrics": {
                    "clock_period_ns": 7.584,
                    "latency_best_cycles": 1057,
                    "interval_min_cycles": 1058,
                    "resources_lut_used": 8741,
                    "resources_ff_used": 21727,
                    "resources_dsp_used": 56,
                    "resources_bram_used": 0,
                },
                "candidates": [
                    {
                        "candidate_index": 1,
                        "verdict": "reject_frequency_threshold",
                        "usefulness_classification": "promising_constraint_violation",
                        "refinement_eligible": True,
                        "minimum_frequency_mhz": 100.0,
                        "fully_verified": False,
                        "metrics": {
                            "clock_period_ns": 63.5,
                            "frequency_mhz": 15.748,
                            "maximum_clock_period_ns": 10.0,
                            "latency_ns": 63458.277,
                            "resources_lut_used": 2468,
                            "resources_ff_used": 1985,
                            "resources_dsp_used": 56,
                            "resources_bram_used": 0,
                        },
                        "deltas_percent": {
                            "resources_lut_used": -71.765,
                            "resources_ff_used": -90.864,
                            "resources_dsp_used": 0.0,
                            "resources_bram_used": 0.0,
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
                "benchmark": "bicg",
                "top_function": "kernel_bicg",
                "baseline": {"source": "baseline.cpp"},
                "output_dir": "out",
            }
        ),
        encoding="utf-8",
    )

    prompt_path = prepare_refinement_prompt(config, 1, 2)
    prompt = prompt_path.read_text(encoding="utf-8")
    strategy = json.loads(
        (output / "candidate_002_strategy.json").read_text(encoding="utf-8")
    )
    feedback = json.loads(
        (output / "candidate_001_feedback.json").read_text(encoding="utf-8")
    )

    assert "Selected strategy: recover_frequency" in prompt
    assert "Estimated frequency: 15.748 MHz" in prompt
    assert "Maximum permitted clock period: 10.0 ns" in prompt
    assert "resources_lut_used: -71.765%" in prompt
    assert "resources_ff_used: -90.864%" in prompt
    assert "critical path" in prompt
    assert "pipelining or parallelism" in prompt
    assert "Do not revert completely to the baseline architecture" in prompt
    assert "Do not perform a large unrelated rewrite" in prompt
    assert strategy["name"] == "recover_frequency"
    assert strategy["source_candidate_index"] == 1
    assert strategy["next_candidate_index"] == 2
    assert strategy["trigger"] == "promising_frequency_constraint_violation"
    assert strategy["preserve"] == ["resources_lut_used", "resources_ff_used"]
    assert strategy["improve"] == ["clock_period_ns", "frequency_mhz", "latency_ns"]
    assert feedback["selected_strategy"]["name"] == "recover_frequency"


def test_recover_frequency_strategy_requires_post_synthesis_evidence() -> None:
    result = check_strategy_compliance(
        "void kernel() {}\n",
        {"name": "recover_frequency", "parameters": {}},
    )

    assert result == {
        "required": False,
        "passed": True,
        "strategy": "recover_frequency",
        "reason": "requires_post_synthesis_evidence",
    }
