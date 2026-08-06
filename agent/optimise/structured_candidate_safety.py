"""Deterministic safety canonicalisation for structured HLS candidates.

The model remains responsible for the architectural rewrite. This module only
normalises source-level directive mistakes that are already forbidden by the
structured prompts and static validator. Legacy strategies and repair workflows
are deliberately untouched.
"""

from __future__ import annotations

import re
from typing import Any


STRUCTURED_ADVISORY_MODE = "advisory"
_BOUNDED_UNROLL_FAMILIES = {
    "bounded_unroll",
    "memory_parallelism",
}


def _matching_brace(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        depth += text[index] == "{"
        depth -= text[index] == "}"
        if depth == 0:
            return index
    return None


def _function_span(text: str, name: str) -> tuple[int, int] | None:
    signature = re.search(
        rf"\b{re.escape(name)}\s*\([^)]*\)\s*\{{",
        text,
        re.MULTILINE,
    )
    if not signature:
        return None
    opening = text.find("{", signature.start())
    closing = _matching_brace(text, opening)
    return (opening, closing) if closing is not None else None


def _function_body(text: str, name: str) -> str:
    span = _function_span(text, name)
    if span is None:
        return ""
    opening, closing = span
    return text[opening + 1 : closing]


def _top_array_parameters(text: str, name: str) -> set[str]:
    signature = re.search(
        rf"\b{re.escape(name)}\s*\((.*?)\)\s*\{{",
        text,
        re.DOTALL,
    )
    if not signature:
        return set()
    parameters: set[str] = set()
    for item in signature.group(1).split(","):
        match = re.search(r"\b([A-Za-z_]\w*)\s*\[[^\]]+\]", item)
        if match:
            parameters.add(match.group(1))
    return parameters


def _replace_complete_interface_partitions(
    source: str,
    *,
    top_function: str,
    factor: int,
) -> tuple[str, list[dict[str, Any]]]:
    interface_arrays = _top_array_parameters(source, top_function)
    if not interface_arrays:
        return source, []

    pattern = re.compile(
        r"^(?P<indent>[ \t]*)#\s*pragma\s+HLS\s+ARRAY_PARTITION\b(?P<args>[^\n]*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    changes: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        arguments = match.group("args")
        variable_match = re.search(
            r"\bvariable\s*=\s*([A-Za-z_]\w*)",
            arguments,
            re.IGNORECASE,
        )
        if (
            variable_match is None
            or variable_match.group(1) not in interface_arrays
            or re.search(r"\bcomplete\b", arguments, re.IGNORECASE) is None
        ):
            return match.group(0)

        variable = variable_match.group(1)
        updated = re.sub(
            r"\bcomplete\b",
            "cyclic",
            arguments,
            count=1,
            flags=re.IGNORECASE,
        ).rstrip()
        if re.search(r"\bfactor\s*=", updated, re.IGNORECASE) is None:
            updated += f" factor={factor}"
        if re.search(r"\bdim\s*=", updated, re.IGNORECASE) is None:
            updated += " dim=1"
        replacement = (
            f"{match.group('indent')}#pragma HLS ARRAY_PARTITION{updated}"
        )
        changes.append(
            {
                "action": "bound_interface_partition",
                "variable": variable,
                "before": match.group(0).strip(),
                "after": replacement.strip(),
            }
        )
        return replacement

    return pattern.sub(replace, source), changes


def _canonicalise_complete_unroll(
    source: str,
    *,
    strategy_name: str,
    factor: int,
) -> tuple[str, list[dict[str, Any]]]:
    pattern = re.compile(
        r"^(?P<indent>[ \t]*)#\s*pragma\s+HLS\s+UNROLL\b(?P<args>[^\n]*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    changes: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        arguments = match.group("args")
        if re.search(r"\bfactor\s*=\s*\d+", arguments, re.IGNORECASE):
            return match.group(0)

        if strategy_name in _BOUNDED_UNROLL_FAMILIES:
            replacement = (
                f"{match.group('indent')}#pragma HLS UNROLL factor={factor}"
            )
            action = "bound_complete_unroll"
        else:
            replacement = ""
            action = "remove_complete_unroll"
        changes.append(
            {
                "action": action,
                "before": match.group(0).strip(),
                "after": replacement.strip() or None,
            }
        )
        return replacement

    return pattern.sub(replace, source), changes


def _remove_pragma_kind(source: str, kind: str) -> tuple[str, int]:
    pattern = re.compile(
        rf"^[ \t]*#\s*pragma\s+HLS\s+{re.escape(kind)}\b[^\n]*\n?",
        re.IGNORECASE | re.MULTILINE,
    )
    return pattern.subn("", source)


def _resolve_dataflow_pipeline_conflict(
    source: str,
    *,
    top_function: str,
    strategy_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    body = _function_body(source, top_function)
    has_dataflow = bool(
        re.search(r"#\s*pragma\s+HLS\s+DATAFLOW\b", body, re.IGNORECASE)
    )
    has_pipeline = bool(
        re.search(r"#\s*pragma\s+HLS\s+PIPELINE\b", body, re.IGNORECASE)
    )
    if not (has_dataflow and has_pipeline):
        return source, []

    loop_count = len(re.findall(r"\bfor\s*\(", body))
    keep_dataflow = (
        strategy_name == "pipeline_dataflow_restructuring"
        and loop_count >= 2
    )
    removed_kind = "PIPELINE" if keep_dataflow else "DATAFLOW"
    updated, count = _remove_pragma_kind(source, removed_kind)
    if count == 0:
        return source, []
    return updated, [
        {
            "action": "resolve_dataflow_pipeline_conflict",
            "removed": removed_kind.lower(),
            "kept": "dataflow" if keep_dataflow else "pipeline",
            "top_loop_count": loop_count,
        }
    ]


def _normalise_loop_header(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _baseline_labelled_loop_header(
    baseline_source: str,
    label: str,
) -> str | None:
    match = re.search(
        rf"^\s*{re.escape(label)}\s*:\s*\n?\s*(for\s*\([^)]*\)\s*\{{)",
        baseline_source,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def _restore_loop_label(
    source: str,
    *,
    baseline_source: str,
    top_function: str,
    label: str,
) -> tuple[str, list[dict[str, Any]]]:
    if re.search(
        rf"^\s*{re.escape(label)}\s*:\s*$",
        source,
        re.MULTILINE,
    ):
        return source, []

    span = _function_span(source, top_function)
    if span is None:
        return source, []
    opening, closing = span
    body = source[opening + 1 : closing]
    loop_matches = list(
        re.finditer(r"\bfor\s*\([^)]*\)\s*\{", body, re.MULTILINE)
    )
    if not loop_matches:
        return source, []

    target: re.Match[str] | None = None
    baseline_header = _baseline_labelled_loop_header(baseline_source, label)
    if baseline_header is not None:
        normalised = _normalise_loop_header(baseline_header)
        exact = [
            match
            for match in loop_matches
            if _normalise_loop_header(match.group(0)) == normalised
        ]
        if len(exact) == 1:
            target = exact[0]

    # A C label does not alter algorithmic behaviour. If the model split or
    # rewrote the original loop so no exact header survives, attach the required
    # audit label to the first top-function loop deterministically.
    if target is None:
        target = loop_matches[0]

    absolute_start = opening + 1 + target.start()
    line_start = source.rfind("\n", 0, absolute_start) + 1
    indentation = re.match(
        r"[ \t]*",
        source[line_start:absolute_start],
    ).group(0)
    insertion = f"{indentation}{label}:\n"
    updated = source[:line_start] + insertion + source[line_start:]
    return updated, [
        {
            "action": "restore_loop_label",
            "label": label,
            "fallback_to_first_loop": baseline_header is None
            or _normalise_loop_header(target.group(0))
            != _normalise_loop_header(baseline_header),
        }
    ]


def canonicalise_structured_candidate(
    source: str,
    *,
    strategy: dict[str, Any] | None,
    top_function: str,
    baseline_source: str,
    validation: dict[str, Any] | None = None,
    target_loop_label: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a safely canonicalised structured candidate and an audit report."""

    if not isinstance(strategy, dict) or (
        strategy.get("compliance_mode") != STRUCTURED_ADVISORY_MODE
    ):
        return source, {
            "applied": False,
            "reason": "not_structured_advisory",
            "changes": [],
        }

    strategy_name = str(strategy.get("name") or "structured_advisory")
    parameters = strategy.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    configured_factor = parameters.get("factor")
    factor = (
        int(configured_factor)
        if isinstance(configured_factor, int)
        and not isinstance(configured_factor, bool)
        and configured_factor > 1
        else 2
    )
    validation_config = validation if isinstance(validation, dict) else {}

    updated = source
    changes: list[dict[str, Any]] = []

    if validation_config.get("reject_dataflow_pipeline_conflict", True):
        updated, applied = _resolve_dataflow_pipeline_conflict(
            updated,
            top_function=top_function,
            strategy_name=strategy_name,
        )
        changes.extend(applied)

    if validation_config.get("reject_pipeline_complete_unroll_conflict", True):
        updated, applied = _canonicalise_complete_unroll(
            updated,
            strategy_name=strategy_name,
            factor=factor,
        )
        changes.extend(applied)

    if validation_config.get("reject_complete_interface_partition", True):
        updated, applied = _replace_complete_interface_partitions(
            updated,
            top_function=top_function,
            factor=factor,
        )
        changes.extend(applied)

    labels = [
        label
        for label in validation_config.get("required_loop_labels", [])
        if isinstance(label, str) and label
    ]
    if (
        validation_config.get("preserve_diagnosed_loop_label", True)
        and isinstance(target_loop_label, str)
        and target_loop_label
        and target_loop_label not in labels
    ):
        labels.append(target_loop_label)
    for label in labels:
        updated, applied = _restore_loop_label(
            updated,
            baseline_source=baseline_source,
            top_function=top_function,
            label=label,
        )
        changes.extend(applied)

    updated = re.sub(r"\n{3,}", "\n\n", updated)
    return updated, {
        "applied": bool(changes),
        "strategy": strategy_name,
        "bounded_factor": factor,
        "changes": changes,
    }
