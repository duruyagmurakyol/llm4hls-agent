from __future__ import annotations

from agent.optimise.refinement_strategy import check_strategy_compliance


BASELINE = """void kernel(double A[8][8], double C[8]) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            C[j] += A[i][j];
        }
    }
}
"""


def _buffered_strategy() -> dict:
    return {
        "name": "buffered_parallelism",
        "compliance_mode": "advisory",
        "parameters": {"allowed_factors": [2, 4, 5, 8]},
    }


def test_buffered_parallelism_requires_memory_and_compute_evidence() -> None:
    candidate = """void kernel(double A[8][8], double C[8]) {
    double row[8];
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=4 dim=1
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) row[j] = A[i][j];
        for (int j = 0; j < 8; ++j) {
#pragma HLS UNROLL factor=4
            C[j] += row[j];
        }
    }
}
"""

    result = check_strategy_compliance(
        candidate,
        _buffered_strategy(),
        baseline=BASELINE,
    )

    assert result["required"] is True
    assert result["passed"] is True
    assert result["reason"] == "buffered_parallelism_evidence_found"
    assert result["observed"]["banked_new_local_arrays"] == ["row"]


def test_buffered_parallelism_rejects_unmatched_banking_only() -> None:
    candidate = """void kernel(double A[8][8], double C[8]) {
    double row[8];
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=4 dim=1
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) row[j] = A[i][j];
        for (int j = 0; j < 8; ++j) C[j] += row[j];
    }
}
"""

    result = check_strategy_compliance(
        candidate,
        _buffered_strategy(),
        baseline=BASELINE,
    )

    assert result["passed"] is False
    assert result["reason"] == "buffered_parallelism_not_realised"
    assert result["observed"]["memory"]["passed"] is True
    assert result["observed"]["bounded_unroll"]["passed"] is False


def test_buffered_parallelism_rejects_top_level_partition_plus_unroll() -> None:
    candidate = """void kernel(double A[8][8], double C[8]) {
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=4 dim=2
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
#pragma HLS UNROLL factor=4
            C[j] += A[i][j];
        }
    }
}
"""

    result = check_strategy_compliance(
        candidate,
        _buffered_strategy(),
        baseline=BASELINE,
    )

    assert result["passed"] is False
    assert result["reason"] == "buffered_parallelism_not_realised"
    assert result["observed"]["new_local_arrays"] == []
    assert result["observed"]["banked_new_local_arrays"] == []
    assert result["observed"]["bounded_unroll"]["passed"] is True


def test_dataflow_pipeline_requires_new_dataflow_pragma() -> None:
    strategy = {
        "name": "dataflow_pipeline",
        "compliance_mode": "advisory",
    }
    pipeline_only = BASELINE.replace(
        "for (int i = 0; i < 8; ++i) {",
        "for (int i = 0; i < 8; ++i) {\n#pragma HLS PIPELINE II=1",
    )
    with_dataflow = BASELINE.replace(
        "    for (int i = 0; i < 8; ++i) {",
        "#pragma HLS DATAFLOW\n    for (int i = 0; i < 8; ++i) {",
    )

    assert check_strategy_compliance(
        pipeline_only,
        strategy,
        baseline=BASELINE,
    )["passed"] is False
    assert check_strategy_compliance(
        with_dataflow,
        strategy,
        baseline=BASELINE,
    )["passed"] is True
