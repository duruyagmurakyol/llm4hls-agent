from __future__ import annotations

from agent.optimise.structured_candidate_safety import (
    canonicalise_structured_candidate,
)


BASELINE = '''#include "vector_add.h"

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


def _strategy(name: str, **parameters: int) -> dict:
    return {
        "name": name,
        "parameters": parameters,
        "compliance_mode": "advisory",
    }


def _validation() -> dict:
    return {
        "reject_complete_interface_partition": True,
        "reject_dataflow_pipeline_conflict": True,
        "reject_pipeline_complete_unroll_conflict": True,
        "required_loop_labels": ["vector_add_loop"],
        "preserve_diagnosed_loop_label": True,
    }


def test_bounded_unroll_candidate_is_made_statically_safe() -> None:
    unsafe = '''#include "vector_add.h"

void vector_add(
    const float a[VECTOR_SIZE],
    const float b[VECTOR_SIZE],
    float c[VECTOR_SIZE]
) {
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=a complete dim=1
#pragma HLS ARRAY_PARTITION variable=b complete
#pragma HLS ARRAY_PARTITION variable=c complete dim=1
    for (int i = 0; i < VECTOR_SIZE; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL
        c[i] = a[i] + b[i];
    }
}
'''

    source, report = canonicalise_structured_candidate(
        unsafe,
        strategy=_strategy("bounded_unroll"),
        top_function="vector_add",
        baseline_source=BASELINE,
        validation=_validation(),
        target_loop_label="vector_add_loop",
    )

    assert report["applied"] is True
    assert "#pragma HLS DATAFLOW" not in source
    assert "#pragma HLS UNROLL factor=2" in source
    assert " complete" not in source
    assert "variable=a cyclic factor=2 dim=1" in source
    assert "variable=b cyclic factor=2 dim=1" in source
    assert "variable=c cyclic factor=2 dim=1" in source
    assert "vector_add_loop:" in source

    actions = [item["action"] for item in report["changes"]]
    assert "resolve_dataflow_pipeline_conflict" in actions
    assert "bound_complete_unroll" in actions
    assert actions.count("bound_interface_partition") == 3
    assert "restore_loop_label" in actions


def test_critical_path_candidate_drops_complete_unroll() -> None:
    unsafe = '''#include "vector_add.h"
void vector_add(const float a[VECTOR_SIZE], const float b[VECTOR_SIZE], float c[VECTOR_SIZE]) {
vector_add_loop:
    for (int i = 0; i < VECTOR_SIZE; ++i) {
#pragma HLS UNROLL
        c[i] = a[i] + b[i];
    }
}
'''

    source, report = canonicalise_structured_candidate(
        unsafe,
        strategy=_strategy("critical_path_restructuring"),
        top_function="vector_add",
        baseline_source=BASELINE,
        validation=_validation(),
        target_loop_label="vector_add_loop",
    )

    assert "#pragma HLS UNROLL" not in source
    assert any(
        item["action"] == "remove_complete_unroll"
        for item in report["changes"]
    )


def test_dataflow_fallback_keeps_dataflow_only_for_multiple_loops() -> None:
    unsafe = '''#include "vector_add.h"
void vector_add(const float a[VECTOR_SIZE], const float b[VECTOR_SIZE], float c[VECTOR_SIZE]) {
#pragma HLS DATAFLOW
stage_one:
    for (int i = 0; i < VECTOR_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        c[i] = a[i];
    }
stage_two:
    for (int i = 0; i < VECTOR_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        c[i] += b[i];
    }
}
'''

    source, report = canonicalise_structured_candidate(
        unsafe,
        strategy=_strategy("pipeline_dataflow_restructuring"),
        top_function="vector_add",
        baseline_source=BASELINE,
        validation={
            "reject_dataflow_pipeline_conflict": True,
            "required_loop_labels": [],
            "preserve_diagnosed_loop_label": False,
        },
    )

    assert "#pragma HLS DATAFLOW" in source
    assert "#pragma HLS PIPELINE" not in source
    change = next(
        item
        for item in report["changes"]
        if item["action"] == "resolve_dataflow_pipeline_conflict"
    )
    assert change["removed"] == "pipeline"
    assert change["kept"] == "dataflow"


def test_non_advisory_strategy_is_not_modified() -> None:
    unsafe = '''#include "vector_add.h"
void vector_add(const float a[VECTOR_SIZE], const float b[VECTOR_SIZE], float c[VECTOR_SIZE]) {
#pragma HLS DATAFLOW
#pragma HLS PIPELINE II=1
    for (int i = 0; i < VECTOR_SIZE; ++i) { c[i] = a[i] + b[i]; }
}
'''
    strategy = {"name": "partial_unroll", "parameters": {"factor": 2}}

    source, report = canonicalise_structured_candidate(
        unsafe,
        strategy=strategy,
        top_function="vector_add",
        baseline_source=BASELINE,
        validation=_validation(),
        target_loop_label="vector_add_loop",
    )

    assert source == unsafe
    assert report == {
        "applied": False,
        "reason": "not_structured_advisory",
        "changes": [],
    }


def test_missing_label_is_not_guessed_when_multiple_loops_are_ambiguous() -> None:
    candidate = '''#include "vector_add.h"
void vector_add(const float a[VECTOR_SIZE], const float b[VECTOR_SIZE], float c[VECTOR_SIZE]) {
    for (int j = 0; j < VECTOR_SIZE; ++j) { c[j] = a[j]; }
    for (int k = 0; k < VECTOR_SIZE; ++k) { c[k] += b[k]; }
}
'''

    source, report = canonicalise_structured_candidate(
        candidate,
        strategy=_strategy("loop_schedule_restructuring"),
        top_function="vector_add",
        baseline_source=BASELINE,
        validation={
            "required_loop_labels": ["vector_add_loop"],
            "preserve_diagnosed_loop_label": True,
        },
        target_loop_label="vector_add_loop",
    )

    assert "vector_add_loop:" not in source
    assert not any(
        item["action"] == "restore_loop_label"
        for item in report["changes"]
    )
