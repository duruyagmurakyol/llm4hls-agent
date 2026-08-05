from __future__ import annotations

from agent.optimise.refinement_strategy import check_strategy_compliance
from agent.optimise.source_text import mask_cpp_comments
from agent.optimise.strategy_canonicalisation import (
    canonicalise_latency_recovery_directives,
)


STRATEGY = {
    "name": "recover_latency_tradeoff",
    "parameters": {"factor": 2},
}

COMPLIANT_SOURCE_WITH_EXPLANATORY_COMMENT = """#include \"bicg.h\"

void kernel_bicg(
    double A[42][38],
    double s[38],
    double q[42],
    double p[38],
    double r[42]) {
    const int n = 42;
    const int m = 38;

    // Apply #pragma HLS PIPELINE II=1 and #pragma HLS UNROLL factor=2.
    for (int j = 0; j < m; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=2
        double s_acc = 0.0;
        for (int i = 0; i < n; i++) {
            s_acc = s_acc + r[i] * A[i][j];
        }
        s[j] = s_acc;
    }

    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=2
        double q_acc = 0.0;
        for (int j = 0; j < m; j++) {
            q_acc = q_acc + A[i][j] * p[j];
        }
        q[i] = q_acc;
    }
}
"""


def test_comment_mask_preserves_offsets_and_literals() -> None:
    source = (
        'const char *value = "// not a comment"; // #pragma HLS UNROLL factor=8\n'
        "/* #pragma HLS PIPELINE II=1\n"
        "   second block-comment line */\n"
        "int result = 0;\n"
    )

    masked = mask_cpp_comments(source)

    assert len(masked) == len(source)
    assert masked.count("\n") == source.count("\n")
    assert '"// not a comment"' in masked
    assert "#pragma HLS" not in masked
    assert "int result = 0;" in masked


def test_explanatory_comment_does_not_create_outer_directives() -> None:
    result = check_strategy_compliance(
        COMPLIANT_SOURCE_WITH_EXPLANATORY_COMMENT,
        STRATEGY,
    )

    assert result["passed"] is True
    assert result["observed"]["outer_pipeline"] is False
    assert result["observed"]["outer_unroll"] is False
    assert result["observed"]["matching_pipeline_unroll_loop"] is True
    assert result["observed"]["loop_unroll_factors"] == [2, 2]


def test_comment_text_does_not_trigger_unnecessary_canonicalisation() -> None:
    updated, count = canonicalise_latency_recovery_directives(
        COMPLIANT_SOURCE_WITH_EXPLANATORY_COMMENT,
        STRATEGY,
    )

    assert count == 0
    assert updated == COMPLIANT_SOURCE_WITH_EXPLANATORY_COMMENT


def test_real_outer_pipeline_is_still_moved_with_comment_present() -> None:
    source = """void kernel(int a[8]) {
    // Request #pragma HLS PIPELINE II=1 and #pragma HLS UNROLL factor=2.
    for (int i = 0; i < 8; ++i) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < 8; ++j) {
            #pragma HLS UNROLL factor=2
            a[j] += i;
        }
    }
}
"""

    updated, count = canonicalise_latency_recovery_directives(source, STRATEGY)
    result = check_strategy_compliance(updated, STRATEGY)

    assert count == 1
    assert result["passed"] is True
    assert updated.count("#pragma HLS PIPELINE II=1") == 2
    assert updated.count("#pragma HLS UNROLL factor=2") == 2
