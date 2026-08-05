"""Canonical failure taxonomy for HLS repair and validation stages."""

from __future__ import annotations

import re
from typing import Literal

FailureStage = Literal["host", "csim", "synthesis", "cosim"]

FAILURE_CLASSES = frozenset(
    {
        "none",
        "syntax_or_compile",
        "missing_header",
        "top_function_mismatch",
        "linkage_or_interface",
        "functional_mismatch",
        "numerical_tolerance",
        "out_of_bounds",
        "csim_timeout",
        "synthesis_unsupported_construct",
        "synthesis_timeout",
        "stream_deadlock",
        "cosim_mismatch",
        "cosim_deadlock",
        "cosim_timeout",
        "tool_report_missing",
        "report_parse",
        "model_generation_error",
        "scope_violation",
        "protected_file_modified",
        "unknown",
    }
)


def infer_failure_stage(output: str) -> FailureStage | None:
    """Infer the tool stage from a raw log when the caller has no stage context."""
    lower = output.lower()
    if any(
        token in lower
        for token in (
            "cosim_design",
            "[cosim ",
            "c/rtl co-simulation",
            "rtl simulation",
            "starting xsim",
        )
    ):
        return "cosim"
    if any(
        token in lower
        for token in (
            "csynth_design",
            "c-synthesis",
            "starting hardware synthesis",
            "synthesizing '",
            "[sched ",
            "[bind ",
        )
    ):
        return "synthesis"
    if any(
        token in lower
        for token in (
            "csim_design",
            "csim start",
            "c-simulation",
            "c simulation",
            "[sim 211-",
        )
    ):
        return "csim"
    return None


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def classify_failure(
    output: str,
    *,
    stage: FailureStage | None = None,
    timed_out: bool = False,
) -> str:
    """Map a raw tool or test log to one stable, stage-aware failure class.

    Specific structural failures deliberately take precedence over broad compiler
    or mismatch wording. This keeps retry prompts and experiment summaries from
    collapsing every failure into a generic compilation or functional bucket.
    """
    lower = output.lower()
    resolved_stage = stage or infer_failure_stage(output)

    stream_deadlock = _contains_any(
        lower,
        (
            "stream deadlock",
            "deadlock detected on stream",
            "blocked read on stream",
            "blocked write on stream",
            "hls::stream is read while empty",
            "hls::stream is written while full",
            "dataflow deadlock",
        ),
    )
    cosim_deadlock = _contains_any(
        lower,
        (
            "deadlock detected",
            "no progress",
            "simulation appears to be stuck",
            "rtl simulation timeout due to no progress",
        ),
    )

    if timed_out or _contains_any(lower, ("timed out", "timeout:", "timeout after")):
        if stream_deadlock:
            return "stream_deadlock"
        if resolved_stage == "csim":
            return "csim_timeout"
        if resolved_stage == "synthesis":
            return "synthesis_timeout"
        if resolved_stage == "cosim":
            return "cosim_deadlock" if cosim_deadlock else "cosim_timeout"
        return "unknown"

    header_reference = bool(
        re.search(r"\.(?:h|hh|hpp|hxx)(?:['\":\s]|$)", lower)
    )
    if _contains_any(
        lower,
        (
            "file not found",
            "no such file or directory",
            "cannot open include file",
            "cannot find include file",
            "header file not found",
        ),
    ) and ("#include" in lower or header_reference):
        return "missing_header"

    if _contains_any(
        lower,
        (
            "top function not found",
            "cannot find top function",
            "failed to find top function",
            "no function named",
            "set_top failed",
            "top-level function not found",
            "top-level function is not defined",
            "top function is not defined",
        ),
    ):
        return "top_function_mismatch"

    if _contains_any(
        lower,
        (
            "undefined reference",
            "multiple definition",
            "conflicting types for",
            "conflicting declaration",
            "no matching function for call",
            "too few arguments to function",
            "too many arguments to function",
            "cannot convert argument",
            "linker command failed",
            "ld returned",
            "mangled name",
            "interface mismatch",
            "port mismatch",
        ),
    ):
        return "linkage_or_interface"

    if _contains_any(
        lower,
        (
            "addresssanitizer",
            "heap-buffer-overflow",
            "stack-buffer-overflow",
            "global-buffer-overflow",
            "index out of bounds",
            "out-of-bounds",
            "out of bounds",
            "buffer overflow",
            "runtime error: index",
            "array subscript",
        ),
    ):
        return "out_of_bounds"

    if stream_deadlock:
        return "stream_deadlock"
    if resolved_stage == "cosim" and cosim_deadlock:
        return "cosim_deadlock"

    if resolved_stage == "synthesis" and _contains_any(
        lower,
        (
            "unsupported construct",
            "not synthesizable",
            "cannot be synthesized",
            "is not supported for synthesis",
            "unsupported for synthesis",
            "dynamic memory allocation is not supported",
            "recursion is not supported",
            "system call is not supported",
            "non-synthesizable",
            "synthesizability check failed",
        ),
    ):
        return "synthesis_unsupported_construct"

    if _contains_any(
        lower,
        (
            "tolerance exceeded",
            "outside tolerance",
            "relative error",
            "absolute error",
            "epsilon",
            "ulp error",
            "error tolerance",
            "numerical mismatch",
        ),
    ):
        return "numerical_tolerance"

    explicit_compile_failure = _contains_any(
        lower,
        (
            "syntax error",
            "compilation error",
            "compile error",
            "failed to compile",
            "error: expected",
            "error: use of undeclared identifier",
            "error: unknown type name",
            "error: no member named",
            "error: invalid operands",
            "error: stray",
        ),
    )
    if explicit_compile_failure:
        return "syntax_or_compile"

    mismatch = _contains_any(
        lower,
        (
            "fail index=",
            "mismatch",
            "wrong result",
            "incorrect result",
            "output mismatch",
            "simulation failed",
        ),
    ) or ("expected=" in lower and "actual=" in lower)
    if mismatch:
        return "cosim_mismatch" if resolved_stage == "cosim" else "functional_mismatch"

    if "error:" in lower and resolved_stage != "cosim":
        return "syntax_or_compile"

    return "unknown"
