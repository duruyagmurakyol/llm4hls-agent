from __future__ import annotations

import pytest

from agent.failures import FAILURE_CLASSES, classify_failure, infer_failure_stage
from agent.tools.command_runner import CommandResult
from agent.tools.validation import from_command


@pytest.mark.parametrize(
    ("output", "stage", "timed_out", "expected"),
    [
        ("kernel.cpp:8: error: expected ';' before '}'", "host", False, "syntax_or_compile"),
        (
            "kernel.cpp:1:10: fatal error: 'kernel.h' file not found\n#include \"kernel.h\"",
            "csim",
            False,
            "missing_header",
        ),
        ("ERROR: top function not found: kernel", "synthesis", False, "top_function_mismatch"),
        ("undefined reference to `kernel(int*, int*)'", "host", False, "linkage_or_interface"),
        ("FAIL index=4 expected=17 actual=-15", "csim", False, "functional_mismatch"),
        (
            "FAIL index=4 expected=0.125 actual=0.126 relative error 0.008 outside tolerance",
            "csim",
            False,
            "numerical_tolerance",
        ),
        (
            "AddressSanitizer: heap-buffer-overflow; runtime error: index 16 out of bounds",
            "csim",
            False,
            "out_of_bounds",
        ),
        ("CSim command timed out after 300 seconds", "csim", True, "csim_timeout"),
        (
            "ERROR: dynamic memory allocation is not supported for synthesis; design is not synthesizable",
            "synthesis",
            False,
            "synthesis_unsupported_construct",
        ),
        (
            "csynth_design timed out after 600 seconds",
            "synthesis",
            True,
            "synthesis_timeout",
        ),
        (
            "ERROR: dataflow deadlock detected on stream fifo; blocked read on stream",
            "cosim",
            False,
            "stream_deadlock",
        ),
        (
            "C/RTL co-simulation: output mismatch expected=17 actual=16",
            "cosim",
            False,
            "cosim_mismatch",
        ),
        (
            "RTL simulation appears to be stuck; deadlock detected; no progress",
            "cosim",
            False,
            "cosim_deadlock",
        ),
        ("An unfamiliar failure without a recognised signature", "host", False, "unknown"),
    ],
)
def test_injected_failures_map_to_meaningful_classes(
    output: str,
    stage: str,
    timed_out: bool,
    expected: str,
) -> None:
    actual = classify_failure(output, stage=stage, timed_out=timed_out)
    assert actual == expected
    assert actual in FAILURE_CLASSES


def test_specific_classes_take_precedence_over_generic_error_wording() -> None:
    assert classify_failure(
        "fatal error: 'vector_add.h' file not found while compiling #include \"vector_add.h\"",
        stage="csim",
    ) == "missing_header"
    assert classify_failure(
        "ERROR: mismatch expected=0.5 actual=0.6; tolerance exceeded",
        stage="cosim",
    ) == "numerical_tolerance"


def test_stage_can_be_inferred_from_vitis_log() -> None:
    log = "INFO: [SIM 211-2] CSIM start\nERROR: timeout after 300 seconds"
    assert infer_failure_stage(log) == "csim"
    assert classify_failure(log, timed_out=True) == "csim_timeout"


def test_from_command_uses_stage_and_timeout_context() -> None:
    result = CommandResult(
        command=("vitis-run",),
        return_code=-15,
        output="command exceeded its time limit",
        cwd=".",
        timed_out=True,
        timeout_seconds=600,
        elapsed_seconds=600.0,
    )
    validation = from_command(result, stage="synthesis")
    assert validation.failure_class == "synthesis_timeout"
    assert validation.evidence
