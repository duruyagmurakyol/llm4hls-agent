from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agent.optimise import generate as generation
from agent.optimise import search_ledger_runtime as runtime
from agent.tools import validation as validation_module
from agent.tools.validation import validate_ppa_candidate


BASELINE = '''#include "vector_add.h"
#define VECTOR_SIZE 1024
void vector_add(
    const float a[VECTOR_SIZE],
    const float b[VECTOR_SIZE],
    float c[VECTOR_SIZE]
) {
vector_add_loop:
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        c[i] = a[i] + b[i];
    }
}
'''

UNSAFE = '''#include "vector_add.h"
#define VECTOR_SIZE 1024
void vector_add(
    const float a[VECTOR_SIZE],
    const float b[VECTOR_SIZE],
    float c[VECTOR_SIZE]
) {
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=a complete dim=1
#pragma HLS ARRAY_PARTITION variable=b complete dim=1
#pragma HLS ARRAY_PARTITION variable=c complete dim=1
    for (int i = 0; i < VECTOR_SIZE; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL
        c[i] = a[i] + b[i];
    }
}
'''


def _config(tmp_path: Path) -> Path:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    baseline = tmp_path / "baseline.cpp"
    baseline.write_text(BASELINE, encoding="utf-8")
    (output_dir / "candidate_001_prompt.txt").write_text(
        "structured prompt\n",
        encoding="utf-8",
    )
    (output_dir / "candidate_001_strategy.json").write_text(
        json.dumps(
            {
                "name": "bounded_unroll",
                "parameters": {"factor": 2},
                "source_candidate_index": 0,
                "next_candidate_index": 1,
                "phase": "explore",
                "compliance_mode": "advisory",
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "output_dir": str(output_dir),
        "baseline": {"source": str(baseline)},
        "top_function": "vector_add",
        "target_loop_label": "vector_add_loop",
        "validation": {
            "constant_loop_tail_bounds": True,
            "reject_complete_interface_partition": True,
            "reject_dataflow_pipeline_conflict": True,
            "reject_pipeline_complete_unroll_conflict": True,
            "required_loop_labels": ["vector_add_loop"],
            "preserve_diagnosed_loop_label": True,
        },
        "model": {
            "provider": "siliconflow",
            "name": "test-model",
            "temperature": 0.0,
            "max_tokens": 128,
        },
        "search_policy": {"mode": "structured_v1"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _bounded_partition_variables(source: str) -> set[str]:
    variables: set[str] = set()
    for line in source.splitlines():
        if "ARRAY_PARTITION" not in line:
            continue
        variable = re.search(
            r"\bvariable\s*=\s*([A-Za-z_]\w*)",
            line,
            re.IGNORECASE,
        )
        if variable is None:
            continue
        if not re.search(r"\bcyclic\b", line, re.IGNORECASE):
            continue
        if not re.search(r"\bfactor\s*=\s*2\b", line, re.IGNORECASE):
            continue
        if not re.search(r"\bdim\s*=\s*1\b", line, re.IGNORECASE):
            continue
        if re.search(r"\bcomplete\b", line, re.IGNORECASE):
            continue
        variables.add(variable.group(1))
    return variables


def test_runtime_canonicalises_before_source_hash_and_static_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validation_module, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)

    def fake_generate(source, candidate_index=1, *, budget=None):
        del source, budget
        candidate = tmp_path / "run" / f"candidate_{candidate_index:03d}.cpp"
        candidate.write_text(UNSAFE, encoding="utf-8")
        return candidate

    monkeypatch.setattr(runtime, "_ORIGINAL_GENERATE", fake_generate)

    candidate = runtime._guarded_generate(config, 1)
    source = candidate.read_text(encoding="utf-8")

    assert "#pragma HLS DATAFLOW" not in source
    assert "#pragma HLS UNROLL factor=2" in source
    assert " complete" not in source
    assert _bounded_partition_variables(source) == {"a", "b", "c"}
    assert "vector_add_loop:" in source

    report_path = (
        tmp_path / "run" / "candidate_001_structured_canonicalisation.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["applied"] is True
    assert len(report["changes"]) >= 6

    static = validate_ppa_candidate(config, 1)
    assert static["passed"] is True, static


def test_runtime_does_not_modify_non_advisory_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generation, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)
    strategy_path = tmp_path / "run" / "candidate_001_strategy.json"
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    strategy.pop("compliance_mode")
    strategy_path.write_text(json.dumps(strategy), encoding="utf-8")

    def fake_generate(source, candidate_index=1, *, budget=None):
        del source, budget
        candidate = tmp_path / "run" / f"candidate_{candidate_index:03d}.cpp"
        candidate.write_text(UNSAFE, encoding="utf-8")
        return candidate

    monkeypatch.setattr(runtime, "_ORIGINAL_GENERATE", fake_generate)
    candidate = runtime._guarded_generate(config, 1)

    assert candidate.read_text(encoding="utf-8") == UNSAFE
    report = json.loads(
        (
            tmp_path
            / "run"
            / "candidate_001_structured_canonicalisation.json"
        ).read_text(encoding="utf-8")
    )
    assert report["applied"] is False
    assert report["reason"] == "not_structured_advisory"
