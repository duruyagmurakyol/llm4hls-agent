"""Minimal evidence-directed strategy choice for PPA refinement."""

from __future__ import annotations

import re
from typing import Any, Iterable

PARTIAL_UNROLL_FACTORS = (8, 4, 2)


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


def _loop_bodies(source: str) -> list[str]:
    bodies: list[str] = []
    for match in re.finditer(r"\bfor\s*\([^)]*\)\s*\{", source):
        opening = source.find("{", match.start())
        closing = _matching_brace(source, opening)
        if closing is not None:
            bodies.append(source[opening + 1 : closing])
    return bodies


def apply_strategy_directives(source: str, strategy: dict[str, Any]) -> str:
    """Deterministically place directives for the one supported strategy."""
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
    """Check the one strategy currently supported by evidence-directed refinement."""
    name = strategy.get("name")
    if name != "partial_unroll":
        return {
            "required": True,
            "passed": False,
            "strategy": name,
            "reason": "unsupported_strategy",
        }

    factor = int(strategy.get("parameters", {}).get("factor", 0))
    all_unroll_pragmas = re.findall(
        r"#\s*pragma\s+HLS\s+UNROLL\b([^\n]*)",
        source,
        re.I,
    )
    complete_unroll = any(
        not re.search(r"\bfactor\s*=\s*\d+", arguments, re.I)
        for arguments in all_unroll_pragmas
    )

    loop_factors: list[int] = []
    loop_pipeline = False
    for body in _loop_bodies(source):
        loop_pipeline = loop_pipeline or bool(
            re.search(r"#\s*pragma\s+HLS\s+PIPELINE\b", body, re.I)
        )
        for arguments in re.findall(
            r"#\s*pragma\s+HLS\s+UNROLL\b([^\n]*)",
            body,
            re.I,
        ):
            match = re.search(r"\bfactor\s*=\s*(\d+)", arguments, re.I)
            if match:
                loop_factors.append(int(match.group(1)))

    first_loop = re.search(r"\bfor\s*\(", source)
    function_prefix = source[: first_loop.start()] if first_loop else source
    outer_pipeline = bool(
        re.search(r"#\s*pragma\s+HLS\s+PIPELINE\b", function_prefix, re.I)
    )
    outer_unroll = bool(
        re.search(r"#\s*pragma\s+HLS\s+UNROLL\b", function_prefix, re.I)
    )

    return {
        "required": True,
        "passed": (
            factor > 0
            and factor in loop_factors
            and not complete_unroll
            and not outer_pipeline
            and not outer_unroll
        ),
        "strategy": name,
        "expected": {
            "factor": factor,
            "directives_inside_loop": True,
            "outer_pipeline": False,
            "outer_unroll": False,
        },
        "observed": {
            "loop_unroll_factors": loop_factors,
            "complete_unroll": complete_unroll,
            "loop_pipeline": loop_pipeline,
            "outer_pipeline": outer_pipeline,
            "outer_unroll": outer_unroll,
        },
    }
