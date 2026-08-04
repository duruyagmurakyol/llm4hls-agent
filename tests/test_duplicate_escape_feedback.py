from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.diagnose import prepare_refinement_prompt


def test_latest_duplicate_forces_structural_escape_from_selected_parent(
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
    (output / "candidate_002_duplicate_check.json").write_text(
        json.dumps({"passed": False, "duplicate_of": 1}),
        encoding="utf-8",
    )
    (output / "candidate_002_strategy.json").write_text(
        json.dumps({"name": "partial_unroll", "parameters": {"factor": 8}}),
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
                "baseline_metrics": {},
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
                        },
                        "deltas_percent": {
                            "resources_lut_used": -71.765,
                            "resources_ff_used": -90.864,
                            "resources_dsp_used": 0.0,
                            "resources_bram_used": 0.0,
                        },
                    },
                    {
                        "candidate_index": 2,
                        "verdict": "reject_duplicate",
                        "duplicate_of": 1,
                    },
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

    prompt_path = prepare_refinement_prompt(config, 1, 3)
    prompt = prompt_path.read_text(encoding="utf-8")
    first_feedback_path = output / "candidate_003_feedback.json"
    first_feedback = json.loads(first_feedback_path.read_text(encoding="utf-8"))

    prepare_refinement_prompt(config, 1, 4)
    second_feedback_path = output / "candidate_004_feedback.json"
    second_feedback = json.loads(second_feedback_path.read_text(encoding="utf-8"))

    assert "Selected strategy: recover_frequency" in prompt
    assert "Duplicate escape requirement" in prompt
    assert "Candidate 002 duplicated candidate 001" in prompt
    assert "change at least one structural mechanism" in prompt
    assert "exact directive placement, loop rewrite, or transformation combination" in prompt
    assert first_feedback_path.is_file()
    assert second_feedback_path.is_file()
    assert first_feedback["previous_candidate_index"] == 1
    assert first_feedback["next_candidate_index"] == 3
    assert second_feedback["previous_candidate_index"] == 1
    assert second_feedback["next_candidate_index"] == 4
    assert first_feedback["selected_strategy"]["name"] == "recover_frequency"
    assert first_feedback["duplicate_escape"]["trigger_candidate_index"] == 2
    assert first_feedback["duplicate_escape"]["duplicate_of_candidate_index"] == 1
    assert first_feedback["duplicate_escape"]["attempted_strategy"] == {
        "name": "partial_unroll",
        "parameters": {"factor": 8},
    }
