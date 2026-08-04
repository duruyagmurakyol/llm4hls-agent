from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.optimise.diagnose import prepare_refinement_prompt, prepare_tradeoff_prompt
from agent.optimise.refinement_strategy import (
    check_strategy_compliance,
    select_tradeoff_strategy,
)
from agent.tools.validation import validate_ppa_candidate


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


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        ([8], 4),
        ([8, 4], 2),
        ([8, 4, 2], None),
    ],
)
def test_partial_unroll_ladder_uses_next_unverified_factor(
    completed: list[int],
    expected: int | None,
) -> None:
    strategy = select_tradeoff_strategy("", completed)

    if expected is None:
        assert strategy is None
    else:
        assert strategy is not None
        assert strategy["parameters"] == {"factor": expected}


def _write_tradeoff_fixture(
    tmp_path: Path,
    *,
    source_index: int,
    summary_candidates: list[dict[str, object]],
    synthesis_log: str = "",
) -> tuple[Path, Path]:
    output = tmp_path / "out"
    output.mkdir(exist_ok=True)
    signature = "void vector_add(int a[16], int b[16], int c[16])"
    (tmp_path / "baseline.cpp").write_text(
        "#include \"vector_add.h\"\n"
        f"{signature} {{\n"
        "  for (int i = 0; i < 16; ++i) c[i] = a[i] + b[i];\n"
        "}\n",
        encoding="utf-8",
    )
    (output / f"candidate_{source_index:03d}.cpp").write_text(
        "#include \"vector_add.h\"\n"
        f"{signature} {{\n"
        "#pragma HLS UNROLL factor=8\n"
        "  for (int i = 0; i < 16; ++i) c[i] = a[i] + b[i];\n"
        "}\n",
        encoding="utf-8",
    )
    log_path = output / f"candidate_{source_index:03d}_synthesis/vitis_synthesis.log"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(synthesis_log, encoding="utf-8")
    metrics = {
        "clock_period_ns": 1.016,
        "latency_best_cycles": 8,
        "interval_min_cycles": 8,
        "resources_lut_used": 441,
        "resources_ff_used": 10,
        "resources_dsp_used": 0,
        "resources_bram_used": 0,
    }
    (output / f"candidate_{source_index:03d}_synthesis.json").write_text(
        json.dumps(
            {
                "passed": True,
                "log_file": f"out/candidate_{source_index:03d}_synthesis/vitis_synthesis.log",
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    (output / "experiment_summary.json").write_text(
        json.dumps(
            {
                "baseline_metrics": metrics,
                "candidates": summary_candidates,
            }
        ),
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
    return config_path, output


def test_tradeoff_prompt_records_selected_partial_unroll_strategy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.diagnose.REPO_ROOT", tmp_path)
    config_path, output = _write_tradeoff_fixture(
        tmp_path,
        source_index=1,
        summary_candidates=[],
        synthesis_log=VITIS_MEMORY_PORT_LOG,
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


def test_verified_factor_eight_advances_tradeoff_prompt_to_factor_four(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.diagnose.REPO_ROOT", tmp_path)
    config_path, output = _write_tradeoff_fixture(
        tmp_path,
        source_index=2,
        summary_candidates=[{"candidate_index": 2, "fully_verified": True}],
    )
    (output / "candidate_002_strategy.json").write_text(
        json.dumps({"name": "partial_unroll", "parameters": {"factor": 8}}),
        encoding="utf-8",
    )

    prompt_path = prepare_tradeoff_prompt(config_path, 2, 3)
    strategy = json.loads(
        (output / "candidate_003_strategy.json").read_text(encoding="utf-8")
    )

    assert "partial loop unrolling with factor 4" in prompt_path.read_text(encoding="utf-8")
    assert strategy["parameters"] == {"factor": 4}
    assert strategy["trigger"] == "partial_unroll_ladder"


def test_unfinished_factor_is_retried_in_refinement_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.diagnose.REPO_ROOT", tmp_path)
    config_path, output = _write_tradeoff_fixture(
        tmp_path,
        source_index=2,
        summary_candidates=[
            {
                "candidate_index": 2,
                "fully_verified": False,
                "verdict": "reject_static",
            }
        ],
    )
    (output / "candidate_002_strategy.json").write_text(
        json.dumps(
            {
                "name": "partial_unroll",
                "parameters": {"factor": 8},
                "reason": "Reduce memory pressure.",
                "required_changes": ["Keep the loop and apply partial unrolling with factor 8."],
                "forbidden_changes": ["Do not completely unroll the loop."],
            }
        ),
        encoding="utf-8",
    )
    (output / "candidate_002_static_validation.json").write_text(
        json.dumps(
            {
                "passed": False,
                "checks": {"strategy_compliant": False},
                "evidence": ["Expected UNROLL factor=8."],
            }
        ),
        encoding="utf-8",
    )
    (output / "baseline_source_target.json").write_text(
        json.dumps({"target_name": "vector_add", "loop_label": None}),
        encoding="utf-8",
    )
    (output / "baseline_source_cause.json").write_text(
        json.dumps(
            {
                "primary_hypothesis": {
                    "category": "memory_port_contention",
                    "interpretation": "Too many concurrent accesses.",
                }
            }
        ),
        encoding="utf-8",
    )

    prompt_path = prepare_refinement_prompt(config_path, 2, 3)
    strategy = json.loads(
        (output / "candidate_003_strategy.json").read_text(encoding="utf-8")
    )

    assert "partial loop unrolling with factor 8" in prompt_path.read_text(encoding="utf-8")
    assert strategy["parameters"] == {"factor": 8}
    assert strategy["trigger"] == "retry_unfinished_strategy"
    assert strategy["retry_of_candidate_index"] == 2


@pytest.mark.parametrize(
    ("pragma", "expected"),
    [
        ("#pragma HLS UNROLL factor=8", True),
        ("#pragma HLS UNROLL factor=4", False),
        ("#pragma HLS UNROLL", False),
    ],
)
def test_partial_unroll_strategy_compliance(pragma: str, expected: bool) -> None:
    strategy = {"name": "partial_unroll", "parameters": {"factor": 8}}
    source = f"void kernel() {{\n{pragma}\n}}\n"

    result = check_strategy_compliance(source, strategy)

    assert result["passed"] is expected


def test_static_validation_rejects_candidate_ignoring_selected_strategy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.tools.validation.REPO_ROOT", tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    signature = "void vector_add(int a[16], int b[16], int c[16])"
    (tmp_path / "baseline.cpp").write_text(
        "#include \"vector_add.h\"\n"
        f"{signature} {{\n"
        "  for (int i = 0; i < 16; ++i) c[i] = a[i] + b[i];\n"
        "}\n",
        encoding="utf-8",
    )
    (output / "candidate_002.cpp").write_text(
        "#include \"vector_add.h\"\n"
        f"{signature} {{\n"
        "#pragma HLS UNROLL factor=4\n"
        "  for (int i = 0; i < 16; ++i) c[i] = a[i] + b[i];\n"
        "}\n",
        encoding="utf-8",
    )
    (output / "candidate_002_strategy.json").write_text(
        json.dumps({"name": "partial_unroll", "parameters": {"factor": 8}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "top_function": "vector_add",
                "baseline": {"source": "baseline.cpp"},
                "output_dir": "out",
                "validation": {"preserve_diagnosed_loop_label": False},
            }
        ),
        encoding="utf-8",
    )

    report = validate_ppa_candidate(config_path, 2)

    assert report["passed"] is False
    assert report["checks"]["strategy_compliant"] is False
    assert report["strategy_compliance"]["expected"] == {"factor": 8}
    assert report["strategy_compliance"]["observed"]["unroll_factors"] == [4]
