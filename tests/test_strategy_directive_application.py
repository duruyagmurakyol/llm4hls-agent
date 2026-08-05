from __future__ import annotations

from agent.optimise.refinement_strategy import (
    apply_strategy_directives,
    check_strategy_compliance,
)


def test_partial_unroll_directives_are_moved_inside_loop() -> None:
    source = """#include \"vector_add.h\"

void vector_add(int a[16], int b[16], int c[16]) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
    for (int i = 0; i < 16; ++i) {
        c[i] = a[i] + b[i];
    }
}
"""
    strategy = {"name": "partial_unroll", "parameters": {"factor": 8}}

    result = apply_strategy_directives(source, strategy)

    loop_start = result.index("for (")
    assert "#pragma HLS PIPELINE" not in result[:loop_start]
    assert "#pragma HLS UNROLL" not in result[:loop_start]
    assert result.count("#pragma HLS PIPELINE II=1") == 1
    assert result.count("#pragma HLS UNROLL factor=8") == 1
    assert check_strategy_compliance(result, strategy)["passed"] is True


def test_unknown_strategy_does_not_modify_source() -> None:
    source = "void kernel() {}\n"

    assert apply_strategy_directives(source, {"name": "unknown"}) == source
