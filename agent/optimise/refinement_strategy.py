"""Minimal evidence-directed strategy choice for PPA refinement."""

from __future__ import annotations

import re
from typing import Any, Iterable

PARTIAL_UNROLL_FACTORS = (8, 4, 2)
LATENCY_RECOVERY_FACTORS = (2, 4, 8)


def select_latency_recovery_factor(
    completed_factors: Iterable[int] = (),
) -> int | None:
    """Return the next bounded factor for latency-recovery exploration."""
    completed = {int(factor) for factor in completed_factors}
    return next(
        (factor for factor in LATENCY_RECOVERY_FACTORS if factor not in completed),
        None,
    )


def select_tradeoff_strategy(
    synthesis_log: str,
    completed_factors: Iterable[int] = (),
) -> dict[str, Any] | None:
    """Choose the next untested partial-unroll factor for the observed bottleneck."""
    completed = {int(factor) for factor in completed_factors}
    lowered = synthesis_log.lower()
    memory_limited = (
        "lower bound of ii" in lowered
        and ("multiple 'load'" in lowered or "multiple 'store'" in lowered)
    )
    full_unroll = "complete unroll" in lowered or (
        "unrolling loop" in lowered and "completely" in lowered
    )
    if not completed and not (memory_limited and full_unroll):
        return None

    factor = next(
        (value for value in PARTIAL_UNROLL_FACTORS if value not in completed),
        None,
    )
    if factor is None:
        return None

    return {
        "name": "partial_unroll",
        "parameters": {"factor": factor},
        "reason": (
            "The previous complete-unroll design was limited by memory-port "
            "contention, so reduce parallel accesses while retaining useful "
            "loop-level parallelism."
        ),
        "required_changes": [
            f"Keep the loop and apply partial unrolling with factor {factor}.",
            (
                "Place the loop directives inside the target loop body, immediately "
                f"after its opening brace: #pragma HLS PIPELINE II=1 followed by "
                f"#pragma HLS UNROLL factor={factor}."
            ),
            "Preserve the existing interface contract.",
        ],
        "forbidden_changes": [
            "Do not place PIPELINE or UNROLL before the target loop.",
            "Do not completely unroll the loop.",
            "Do not pipeline the entire top function.",
            "Do not completely partition top-level interface arrays.",
        ],
    }


def _matching_brace(source: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(source)):
        depth += source[index] == "{"
        depth -= source[index] == "}"
        if depth == 0:
            return index
    return None


def _loop_spans(source: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\bfor\s*\([^)]*\)\s*\{", source):
        opening = source.find("{", match.start())
        closing = _matching_brace(source, opening)
        if closing is not None:
            spans.append((opening, closing))
    return spans


def _innermost_loop(
    spans: list[tuple[int, int]],
    position: int,
) -> int | None:
    containing = [
        (index, opening)
        for index, (opening, closing) in enumerate(spans)
        if opening < position < closing
    ]
    if not containing:
        return None
    return max(containing, key=lambda item: item[1])[0]


def apply_strategy_directives(source: str, strategy: dict[str, Any]) -> str:
    """Deterministically place directives for the supported source rewrite."""
    if strategy.get("name") != "partial_unroll":
        return source

    factor = int(strategy.get("parameters", {}).get("factor", 0))
    if factor <= 0:
        return source

    cleaned = re.sub(
        r"^[ \t]*#\s*pragma\s+HLS\s+(?:PIPELINE|UNROLL)\b[^\n]*\n?",
        "",
        source,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    loop = re.search(r"\bfor\s*\([^)]*\)\s*\{", cleaned)
    if not loop:
        return cleaned

    opening = cleaned.find("{", loop.start())
    line_start = cleaned.rfind("\n", 0, loop.start()) + 1
    loop_indent = re.match(r"[ \t]*", cleaned[line_start:loop.start()]).group(0)
    directive_indent = loop_indent + "    "
    directives = (
        f"\n{directive_indent}#pragma HLS PIPELINE II=1"
        f"\n{directive_indent}#pragma HLS UNROLL factor={factor}"
    )
    return cleaned[: opening + 1] + directives + cleaned[opening + 1 :]


def check_strategy_compliance(source: str, strategy: dict[str, Any]) -> dict[str, Any]:
    """Check source-enforceable strategies before running Vitis."""
    name = strategy.get("name")
    factor = int(strategy.get("parameters", {}).get("factor", 0))
    if name == "recover_frequency" or (
        name == "recover_latency_tradeoff" and factor <= 0
    ):
        return {
            "required": False,
            "passed": True,
            "strategy": name,
            "reason": "requires_post_synthesis_evidence",
        }
    if name not in {"partial_unroll", "recover_latency_tradeoff"}:
        return {
            "required": True,
            "passed": False,
            "strategy": name,
            "reason": "unsupported_strategy",
        }

    spans = _loop_spans(source)
    loop_directives = [
        {
            "pipeline": False,
            "unroll_factors": [],
            "complete_unroll": False,
        }
        for _ in spans
    ]
    outer_pipeline = False
    outer_unroll = False
    loop_unroll_factors: list[int] = []
    outer_unroll_factors: list[int] = []
    complete_unroll = False

    pragma_pattern = re.compile(
        r"#\s*pragma\s+HLS\s+(PIPELINE|UNROLL)\b([^\n]*)",
        re.IGNORECASE,
    )
    for pragma in pragma_pattern.finditer(source):
        kind = pragma.group(1).upper()
        arguments = pragma.group(2)
        loop_index = _innermost_loop(spans, pragma.start())

        if kind == "PIPELINE":
            if loop_index is None:
                outer_pipeline = True
            else:
                loop_directives[loop_index]["pipeline"] = True
            continue

        factor_match = re.search(r"\bfactor\s*=\s*(\d+)", arguments, re.I)
        pragma_factor = int(factor_match.group(1)) if factor_match else None

        if loop_index is None:
            outer_unroll = True
            if pragma_factor is not None:
                outer_unroll_factors.append(pragma_factor)
            continue

        if pragma_factor is None:
            complete_unroll = True
            loop_directives[loop_index]["complete_unroll"] = True
        else:
            loop_unroll_factors.append(pragma_factor)
            loop_directives[loop_index]["unroll_factors"].append(pragma_factor)

    matching_loop = any(
        directives["pipeline"] is True
        and factor in directives["unroll_factors"]
        for directives in loop_directives
    )
    loop_pipeline = any(
        directives["pipeline"] is True
        for directives in loop_directives
    )
    factor_inside_loop = factor in loop_unroll_factors
    requires_matching_pipeline = name == "recover_latency_tradeoff"

    return {
        "required": True,
        "passed": (
            factor > 0
            and (
                matching_loop
                if requires_matching_pipeline
                else factor_inside_loop
            )
            and not complete_unroll
            and not outer_pipeline
            and not outer_unroll
        ),
        "strategy": name,
        "expected": {
            "factor": factor,
            "factor_inside_loop": True,
            "pipeline_and_unroll_on_same_loop": requires_matching_pipeline,
            "outer_pipeline": False,
            "outer_unroll": False,
        },
        "observed": {
            "loop_unroll_factors": loop_unroll_factors,
            "outer_unroll_factors": outer_unroll_factors,
            "complete_unroll": complete_unroll,
            "loop_pipeline": loop_pipeline,
            "matching_pipeline_unroll_loop": matching_loop,
            "loop_directives": loop_directives,
            "outer_pipeline": outer_pipeline,
            "outer_unroll": outer_unroll,
        },
    }
