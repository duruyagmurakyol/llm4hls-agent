from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.diagnose import prepare_tradeoff_prompt
from agent.optimise.refinement_strategy import select_tradeoff_strategy


VITIS_MEMORY_PORT_LOG = """
INFO: Loop is marked as complete unroll implied by the pipeline pragma.
INFO: Unrolling loop completely with a factor of 16.
WARNING: Lower bound of II is 8 due to multiple 'load' operations on array 'a'.
"""


def test_selects_partial_unroll_for_memory_limited_complete_unroll() -> None:
    strategy = select_tradeoff_strategy(VITIS_MEMORY_PORT_LOG)

    assert strategy is not None
    assert strategy["name"] == "partial_unroll"
    assert strategy["parameters"] == {"factor": 8}


def test_tradeoff_prompt_records_selected_partial_unroll_strategy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.diagnose.REPO_ROOT", tmp_path)
    output = tmp_path / "out"
    output.mkdir()

    (tmp_path / "baseline.cpp").write_text(
        "void vector_add(int a[16], int b[16], int c[16]) {}\n",
        encoding="utf-8",
    )
    (output / "candidate_001.cpp").write_text(
        "void vector_add(int a[16], int b[16], int c[16]) {\n"
        "#pragma HLS PIPELINE II=1\n"
        "  for (int i = 0; i < 16; ++i) c[i] = a[i] + b[i];\n"
        "}\n",
        encoding="utf-8",
    )
    log_path = output / "candidate_001_synthesis/vitis_synthesis.log"
    log_path.parent.mkdir()
    log_path.write_text(VITIS_MEMORY_PORT_LOG, encoding="utf-8")

    metrics = {
        "clock_period_ns": 1.016,
        "latency_best_cycles": 8,
        "interval_min_cycles": 8,
        "resources_lut_used": 441,
        "resources_ff_used": 10,
        "resources_dsp_used": 0,
        "resources_bram_used": 0,
    }
    (output / "candidate_001_synthesis.json").write_text(
        json.dumps(
            {
                "passed": True,
                "log_file": "out/candidate_001_synthesis/vitis_synthesis.log",
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    (output / "experiment_summary.json").write_text(
        json.dumps({"baseline_metrics": metrics}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "benchmark": "vector_add",
                "top_function": "vector_add",
                "baseline": {"source": "baseline.cpp"},
                "output_dir": "out",
            }
        ),
        encoding="utf-8",
    )

    prompt_path = prepare_tradeoff_prompt(config_path, 1, 2)
    prompt = prompt_path.read_text(encoding="utf-8")
    strategy = json.loads(
        (output / "candidate_002_strategy.json").read_text(encoding="utf-8")
    )

    assert "partial loop unrolling with factor 8" in prompt
    assert "Do not completely unroll the loop" in prompt
    assert "Do not completely partition top-level interface arrays" in prompt
    assert strategy["name"] == "partial_unroll"
    assert strategy["parameters"] == {"factor": 8}
    assert strategy["trigger"] == "memory_port_limited_complete_unroll"
