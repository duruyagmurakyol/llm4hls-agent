from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.optimise.generate import generate_candidate
from agent.optimise.refinement_strategy import check_strategy_compliance
from agent.optimise.strategy_canonicalisation import (
    canonicalise_latency_recovery_directives,
)


STRATEGY = {
    "name": "recover_latency_tradeoff",
    "parameters": {"factor": 2},
    "source_candidate_index": 7,
    "next_candidate_index": 8,
}

MISPLACED_SOURCE = """#include "bicg.h"

void kernel_bicg(
    double A[42][38],
    double s[38],
    double q[42],
    double p[38],
    double r[42]) {
#pragma HLS top name = kernel_bicg

    const int n = 42;
    const int m = 38;

    for (int j = 0; j < m; j++) {
        #pragma HLS PIPELINE II=1
        double s_acc = 0.0;
        for (int i = 0; i < n; i++) {
            #pragma HLS UNROLL factor=2
            s_acc = s_acc + r[i] * A[i][j];
        }
        s[j] = s_acc;
    }

    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        double q_acc = 0.0;
        for (int j = 0; j < m; j++) {
            #pragma HLS UNROLL factor=2
            q_acc = q_acc + A[i][j] * p[j];
        }
        q[i] = q_acc;
    }
}
"""

CANONICAL_SOURCE = """#include "bicg.h"

void kernel_bicg(
    double A[42][38],
    double s[38],
    double q[42],
    double p[38],
    double r[42]) {
#pragma HLS top name = kernel_bicg

    const int n = 42;
    const int m = 38;

    for (int j = 0; j < m; j++) {
        double s_acc = 0.0;
        for (int i = 0; i < n; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=2
            s_acc = s_acc + r[i] * A[i][j];
        }
        s[j] = s_acc;
    }

    for (int i = 0; i < n; i++) {
        double q_acc = 0.0;
        for (int j = 0; j < m; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=2
            q_acc = q_acc + A[i][j] * p[j];
        }
        q[i] = q_acc;
    }
}
"""


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        input_tokens=100,
        output_tokens=200,
        total_tokens=300,
        latency_seconds=0.5,
    )


def test_candidate_008_directives_are_canonicalised_before_validation() -> None:
    before = check_strategy_compliance(MISPLACED_SOURCE, STRATEGY)
    assert before["passed"] is False
    assert before["observed"]["matching_pipeline_unroll_loop"] is False

    updated, count = canonicalise_latency_recovery_directives(
        MISPLACED_SOURCE,
        STRATEGY,
    )

    assert count == 2
    assert updated == CANONICAL_SOURCE
    after = check_strategy_compliance(updated, STRATEGY)
    assert after["passed"] is True
    assert after["observed"]["matching_pipeline_unroll_loop"] is True
    assert after["observed"]["loop_unroll_factors"] == [2, 2]


def test_already_compliant_latency_recovery_source_is_unchanged() -> None:
    updated, count = canonicalise_latency_recovery_directives(
        CANONICAL_SOURCE,
        STRATEGY,
    )

    assert count == 0
    assert updated == CANONICAL_SOURCE


def test_model_generation_persists_canonical_source_and_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.generate.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "agent.optimise.generate.complete",
        lambda **kwargs: _response(MISPLACED_SOURCE),
    )

    output = tmp_path / "out"
    output.mkdir()
    (output / "candidate_008_prompt.txt").write_text(
        "Recover the latency trade-off.",
        encoding="utf-8",
    )
    (output / "candidate_008_strategy.json").write_text(
        json.dumps(STRATEGY),
        encoding="utf-8",
    )
    config_path = tmp_path / "task.json"
    config_path.write_text(
        json.dumps(
            {
                "top_function": "kernel_bicg",
                "output_dir": "out",
                "model": {
                    "provider": "siliconflow",
                    "name": "test-model",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
            }
        ),
        encoding="utf-8",
    )

    candidate_path = generate_candidate(config_path, candidate_index=8)

    assert candidate_path.read_text(encoding="utf-8") == CANONICAL_SOURCE
    assert (
        output / "candidate_008_model_response.txt"
    ).read_text(encoding="utf-8") == MISPLACED_SOURCE
    metadata = json.loads(
        (output / "candidate_008_model_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["strategy_directives_applied"] is False
    assert metadata["strategy_directives_canonicalised"] is True
    assert metadata["strategy_directive_canonicalisations"] == 2
