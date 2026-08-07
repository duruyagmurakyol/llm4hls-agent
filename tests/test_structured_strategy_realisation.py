from agent.optimise.refinement_strategy import check_strategy_compliance


BASELINE = """#include <stdint.h>
void kernel(int a[8], int b[8]) {
    for (int i = 0; i < 8; ++i) {
        b[i] = a[i] + 1;
    }
}
"""


def advisory(name: str, **parameters):
    return {
        "name": name,
        "parameters": parameters,
        "compliance_mode": "advisory",
    }


def test_comment_only_change_does_not_realise_structured_strategy():
    candidate = BASELINE.replace(
        "    for (int i = 0; i < 8; ++i) {",
        "    // optimisation attempt\n    for (int i = 0; i < 8; ++i) {",
    )
    result = check_strategy_compliance(
        candidate,
        advisory("memory_parallelism"),
        baseline=BASELINE,
    )
    assert result["required"] is True
    assert result["passed"] is False
    assert result["reason"] == "no_semantic_change"


def test_bounded_unroll_requires_allowed_factor():
    candidate = BASELINE.replace(
        "    for (int i = 0; i < 8; ++i) {",
        "    for (int i = 0; i < 8; ++i) {\n        #pragma HLS UNROLL factor=2",
    )
    result = check_strategy_compliance(
        candidate,
        advisory("bounded_unroll", allowed_factors=[2, 4]),
        baseline=BASELINE,
    )
    assert result["passed"] is True
    assert result["observed"]["observed_factors"] == [2]


def test_memory_parallelism_requires_memory_structure_evidence():
    candidate = BASELINE.replace(
        "void kernel(int a[8], int b[8]) {",
        "void kernel(int a[8], int b[8]) {\n#pragma HLS ARRAY_PARTITION variable=a cyclic factor=2",
    )
    result = check_strategy_compliance(
        candidate,
        advisory("memory_parallelism"),
        baseline=BASELINE,
    )
    assert result["passed"] is True
    assert result["observed"]["new_memory_pragmas"]


def test_pipeline_family_requires_new_pipeline_or_dataflow_pragma():
    candidate = BASELINE.replace(
        "    for (int i = 0; i < 8; ++i) {",
        "    for (int i = 0; i < 8; ++i) {\n        #pragma HLS PIPELINE II=1",
    )
    result = check_strategy_compliance(
        candidate,
        advisory("pipeline_dataflow_restructuring"),
        baseline=BASELINE,
    )
    assert result["passed"] is True


def test_loop_schedule_family_requires_changed_loop_headers():
    candidate = BASELINE.replace(
        "for (int i = 0; i < 8; ++i)",
        "for (int i = 0; i < 8; i += 2)",
    )
    result = check_strategy_compliance(
        candidate,
        advisory("loop_schedule_restructuring"),
        baseline=BASELINE,
    )
    assert result["passed"] is True


def test_critical_path_family_accepts_real_executable_change():
    candidate = BASELINE.replace("a[i] + 1", "(a[i] + 1)")
    result = check_strategy_compliance(
        candidate,
        advisory("critical_path_restructuring"),
        baseline=BASELINE,
    )
    assert result["passed"] is True
