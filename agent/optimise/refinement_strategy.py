"""Minimal evidence-directed strategy choice for PPA refinement."""

from __future__ import annotations

import re
from typing import Any, Iterable

from agent.optimise.source_text import mask_cpp_comments

PARTIAL_UNROLL_FACTORS = (8, 4, 2)
LATENCY_RECOVERY_FACTORS = (2, 4, 8)
STRUCTURED_ADVISORY_STRATEGIES = {
    "critical_path_restructuring",
    "bounded_unroll",
    "memory_parallelism",
    "loop_schedule_restructuring",
    "pipeline_dataflow_restructuring",
}


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


def _normalised_semantic_source(source: str) -> str:
    """Ignore comments and formatting while preserving executable/HLS tokens."""

    return re.sub(r"\s+", "", mask_cpp_comments(source))


def _semantic_change(source: str, baseline: str | None) -> bool | None:
    if baseline is None:
        return None
    return _normalised_semantic_source(source) != _normalised_semantic_source(baseline)


def _loop_headers(source: str) -> list[str]:
    analysed = mask_cpp_comments(source)
    return [
        re.sub(r"\s+", "", match.group(1))
        for match in re.finditer(r"\bfor\s*\(([^)]*)\)", analysed)
    ]


def _normalised_pragmas(source: str, kinds: str) -> set[str]:
    analysed = mask_cpp_comments(source)
    pattern = re.compile(
        rf"#\s*pragma\s+HLS\s+(?:{kinds})\b[^\n]*",
        re.IGNORECASE,
    )
    return {
        re.sub(r"\s+", "", match.group(0)).casefold()
        for match in pattern.finditer(analysed)
    }


def _bounded_unroll_evidence(source: str, strategy: dict[str, Any]) -> dict[str, Any]:
    allowed = strategy.get("parameters", {}).get("allowed_factors") or [2, 4]
    allowed_factors = {
        int(value)
        for value in allowed
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    }
    analysed = mask_cpp_comments(source)
    observed = [
        int(match.group(1))
        for match in re.finditer(
            r"#\s*pragma\s+HLS\s+UNROLL\b[^\n]*\bfactor\s*=\s*(\d+)",
            analysed,
            re.IGNORECASE,
        )
    ]
    return {
        "passed": any(factor in allowed_factors for factor in observed),
        "allowed_factors": sorted(allowed_factors),
        "observed_factors": observed,
    }


def _local_array_declarations(source: str) -> set[str]:
    """Return normalised local-looking array declarations as weak buffer evidence."""

    analysed = mask_cpp_comments(source)
    pattern = re.compile(
        r"\b(?:static\s+)?(?:const\s+)?[A-Za-z_]\w*(?:::\w+)?(?:\s*<[^;{}]+>)?"
        r"(?:\s+[A-Za-z_]\w*)+\s+[A-Za-z_]\w*\s*(?:\[[^\]]+\])+\s*(?:=\s*\{[^;]*\})?\s*;",
        re.MULTILINE,
    )
    return {
        re.sub(r"\s+", "", match.group(0))
        for match in pattern.finditer(analysed)
    }


def _memory_parallelism_evidence(source: str, baseline: str | None) -> dict[str, Any]:
    candidate_pragmas = _normalised_pragmas(
        source,
        "ARRAY_PARTITION|ARRAY_RESHAPE|BIND_STORAGE|STREAM",
    )
    baseline_pragmas = (
        _normalised_pragmas(
            baseline,
            "ARRAY_PARTITION|ARRAY_RESHAPE|BIND_STORAGE|STREAM",
        )
        if baseline is not None
        else set()
    )
    new_pragmas = sorted(candidate_pragmas - baseline_pragmas)

    candidate_arrays = _local_array_declarations(source)
    baseline_arrays = _local_array_declarations(baseline) if baseline is not None else set()
    new_arrays = sorted(candidate_arrays - baseline_arrays)
    return {
        "passed": bool(new_pragmas or new_arrays),
        "new_memory_pragmas": new_pragmas,
        "new_local_array_declarations": new_arrays,
    }


def _pipeline_dataflow_evidence(source: str, baseline: str | None) -> dict[str, Any]:
    candidate = _normalised_pragmas(source, "PIPELINE|DATAFLOW")
    previous = (
        _normalised_pragmas(baseline, "PIPELINE|DATAFLOW")
        if baseline is not None
        else set()
    )
    added = sorted(candidate - previous)
    return {
        "passed": bool(added),
        "new_pipeline_or_dataflow_pragmas": added,
    }


def _advisory_strategy_compliance(
    source: str,
    strategy: dict[str, Any],
    baseline: str | None,
) -> dict[str, Any]:
    """Require observable source evidence for the bounded structured families."""

    name = strategy.get("name")
    semantic_change = _semantic_change(source, baseline)
    if name not in STRUCTURED_ADVISORY_STRATEGIES:
        return {
            "required": False,
            "passed": True,
            "strategy": name,
            "reason": "advisory_strategy_not_source_enforced",
        }

    if baseline is None:
        return {
            "required": False,
            "passed": True,
            "strategy": name,
            "reason": "structured_strategy_requires_baseline_for_source_audit",
        }

    if semantic_change is not True:
        return {
            "required": True,
            "passed": False,
            "strategy": name,
            "reason": "no_semantic_change",
            "observed": {"semantic_change": semantic_change},
        }

    if name == "critical_path_restructuring":
        return {
            "required": True,
            "passed": True,
            "strategy": name,
            "reason": "executable_dependency_structure_changed",
            "observed": {"semantic_change": True},
        }

    if name == "bounded_unroll":
        evidence = _bounded_unroll_evidence(source, strategy)
        return {
            "required": True,
            "passed": bool(evidence["passed"]),
            "strategy": name,
            "reason": (
                "bounded_unroll_evidence_found"
                if evidence["passed"]
                else "bounded_unroll_not_realised"
            ),
            "observed": evidence,
        }

    if name == "memory_parallelism":
        evidence = _memory_parallelism_evidence(source, baseline)
        return {
            "required": True,
            "passed": bool(evidence["passed"]),
            "strategy": name,
            "reason": (
                "memory_parallelism_evidence_found"
                if evidence["passed"]
                else "memory_parallelism_not_realised"
            ),
            "observed": evidence,
        }

    if name == "loop_schedule_restructuring":
        baseline_headers = _loop_headers(baseline)
        candidate_headers = _loop_headers(source)
        passed = candidate_headers != baseline_headers
        return {
            "required": True,
            "passed": passed,
            "strategy": name,
            "reason": (
                "loop_schedule_changed"
                if passed
                else "loop_schedule_not_realised"
            ),
            "observed": {
                "baseline_loop_headers": baseline_headers,
                "candidate_loop_headers": candidate_headers,
            },
        }

    evidence = _pipeline_dataflow_evidence(source, baseline)
    return {
        "required": True,
        "passed": bool(evidence["passed"]),
        "strategy": name,
        "reason": (
            "pipeline_or_dataflow_evidence_found"
            if evidence["passed"]
            else "pipeline_dataflow_not_realised"
        ),
        "observed": evidence,
    }


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


def check_strategy_compliance(
    source: str,
    strategy: dict[str, Any],
    *,
    baseline: str | None = None,
) -> dict[str, Any]:
    """Check source-enforceable strategies before running Vitis.

    Structured model-guided families carry ``compliance_mode=advisory`` but are
    still required to leave observable source evidence.  This prevents an LLM
    from satisfying a search slot with comments, whitespace, or an unrelated
    edit while retaining compatibility for older advisory strategy metadata.
    """
    name = strategy.get("name")
    if strategy.get("compliance_mode") == "advisory":
        return _advisory_strategy_compliance(source, strategy, baseline)

    factor = int(strategy.get("parameters", {}).get("factor", 0))
    if name in {
        "recover_frequency",
        "recover_resource_limits",
        "recover_resource_frequency_balance",
    } or (
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

    analysed_source = mask_cpp_comments(source)
    spans = _loop_spans(analysed_source)
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
    for pragma in pragma_pattern.finditer(analysed_source):
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
