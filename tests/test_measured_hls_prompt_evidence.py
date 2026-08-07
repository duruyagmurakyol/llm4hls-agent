from __future__ import annotations

import json
from pathlib import Path

from agent.analysis.hls_bottleneck_analyzer import analyse
from scripts.run_ppa_optimisation import generate_optimisation_prompt


def test_vitis_core_ram_lower_bound_is_memory_port_contention() -> None:
    evidence = {
        "warnings": [
            "WARNING: [HLS 200-448] Lower bound of II is 19 due to multiple "
            "'load' operation 64 bit ('A_load') on array 'A' accessing core:RAM:A"
        ],
        "loops": [
            {
                "name": "VITIS_LOOP_18_2",
                "achieved_ii": 19,
                "target_ii": None,
                "latency_cycles": 899,
                "pipelined": True,
            }
        ],
        "top_function": {"interval_cycles": 981},
        "constraints": {"interface_frozen": False},
        "clock": {},
        "resources": {},
    }

    result = analyse(evidence)
    primary = result["primary_diagnosis"]

    assert primary["category"] == "memory_port_contention"
    assert primary["target"] == "A"
    assert "reported_ii_lower_bound=19" in primary["evidence"]
    assert "contended_arrays=A" in primary["evidence"]


def test_optimisation_prompt_contains_nested_measured_hls_evidence(tmp_path: Path) -> None:
    source = tmp_path / "kernel.cpp"
    source.write_text(
        "void kernel(double A[42][38]) {\n"
        "    for (int i = 0; i < 42; ++i) { A[i][0] += 1.0; }\n"
        "}\n",
        encoding="utf-8",
    )

    diagnosis_path = tmp_path / "diagnosis.json"
    diagnosis_path.write_text("{}\n", encoding="utf-8")

    source_target_path = tmp_path / "source_target.json"
    source_target_path.write_text(
        json.dumps(
            {
                "target_name": "kernel_Pipeline_VITIS_LOOP_18_2",
                "loop_label": None,
                "source_file": "kernel.cpp",
                "region_start_line": 2,
                "region_end_line": 2,
                "source_excerpt": "   2:     for (int i = 0; i < 42; ++i) { A[i][0] += 1.0; }",
                "diagnosis": {
                    "function": "kernel_Pipeline_VITIS_LOOP_18_2",
                    "latency_cycles": 899,
                    "interval_cycles": 981,
                    "max_achieved_ii": 19,
                    "primary_diagnosis": {
                        "category": "memory_port_contention",
                        "target": "A",
                        "confidence": 0.98,
                        "evidence": [
                            "reported_ii_lower_bound=19",
                            "contended_arrays=A",
                        ],
                        "recommended_transformations": [
                            "bank or partition a bounded local buffer for the contended access pattern"
                        ],
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    source_cause_path = tmp_path / "source_cause.json"
    source_cause_path.write_text(
        json.dumps(
            {
                "primary_hypothesis": {
                    "category": "memory_access_or_port_pressure",
                    "confidence": 0.7,
                    "interpretation": "Memory banking may limit parallelism.",
                    "evidence": {"arrays": ["A"]},
                },
                "alternative_hypotheses": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "benchmark": "bicg_u55c",
        "top_function": "kernel",
        "output_dir": "out",
        "prompt_constraints": [],
    }
    (tmp_path / "out").mkdir()

    prompt_path = generate_optimisation_prompt(
        config,
        tmp_path,
        diagnosis_path,
        source_target_path,
        source_cause_path,
    )
    prompt = prompt_path.read_text(encoding="utf-8")

    assert "Measured HLS evidence (authoritative):" in prompt
    assert "Diagnosis category: memory_port_contention" in prompt
    assert "Diagnosis target: A" in prompt
    assert "Achieved II: 19" in prompt
    assert "Target-region latency: 899 cycles" in prompt
    assert "reported_ii_lower_bound=19" in prompt
    assert "contended_arrays=A" in prompt
    assert "Report diagnosis: unknown" not in prompt
