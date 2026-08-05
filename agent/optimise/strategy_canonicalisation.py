"""Canonicalise narrow source-level realisations of controlled HLS strategies."""

from __future__ import annotations

import re
from typing import Any

_PRAGMA_PATTERN = re.compile(
    r"#\s*pragma\s+HLS\s+(PIPELINE|UNROLL)\b([^\n]*)",
    re.IGNORECASE,
)
_LOOP_PATTERN = re.compile(r"\bfor\s*\([^)]*\)\s*\{")


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
    for match in _LOOP_PATTERN.finditer(source):
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


def _line_span(source: str, start: int, end: int) -> tuple[int, int]:
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", end)
    return line_start, len(source) if line_end < 0 else line_end + 1


def canonicalise_latency_recovery_directives(
    source: str,
    strategy: dict[str, Any] | None,
) -> tuple[str, int]:
    """Move a nearest-ancestor PIPELINE onto each requested unrolled loop.

    The transformation is intentionally narrow. It only applies to
    ``recover_latency_tradeoff``, only for the configured bounded factor, and
    only when the target loop lacks its own PIPELINE while a containing loop
    carries one. Existing same-loop placements and unrelated pragmas remain
    unchanged.
    """
    if not strategy or strategy.get("name") != "recover_latency_tradeoff":
        return source, 0

    factor = (strategy.get("parameters") or {}).get("factor")
    if isinstance(factor, bool) or not isinstance(factor, int) or factor <= 0:
        return source, 0

    spans = _loop_spans(source)
    if not spans:
        return source, 0

    pragmas: list[dict[str, Any]] = []
    for match in _PRAGMA_PATTERN.finditer(source):
        factor_match = re.search(
            r"\bfactor\s*=\s*(\d+)",
            match.group(2),
            re.IGNORECASE,
        )
        pragmas.append(
            {
                "kind": match.group(1).upper(),
                "factor": int(factor_match.group(1)) if factor_match else None,
                "loop_index": _innermost_loop(spans, match.start()),
                "match": match,
            }
        )

    pipelines_by_loop: dict[int, list[dict[str, Any]]] = {}
    for pragma in pragmas:
        if pragma["kind"] == "PIPELINE" and pragma["loop_index"] is not None:
            pipelines_by_loop.setdefault(pragma["loop_index"], []).append(pragma)

    edits: list[tuple[int, int, str]] = []
    used_pipelines: set[int] = set()

    for pragma in pragmas:
        if (
            pragma["kind"] != "UNROLL"
            or pragma["factor"] != factor
            or pragma["loop_index"] is None
        ):
            continue

        loop_index = int(pragma["loop_index"])
        if pipelines_by_loop.get(loop_index):
            continue

        opening, closing = spans[loop_index]
        ancestors = [
            (index, ancestor_opening)
            for index, (ancestor_opening, ancestor_closing) in enumerate(spans)
            if (
                ancestor_opening < opening
                and closing < ancestor_closing
                and pipelines_by_loop.get(index)
            )
        ]
        if not ancestors:
            continue

        ancestor_index = max(ancestors, key=lambda item: item[1])[0]
        pipeline = next(
            (
                item
                for item in pipelines_by_loop[ancestor_index]
                if item["match"].start() not in used_pipelines
            ),
            None,
        )
        if pipeline is None:
            continue

        pipeline_match = pipeline["match"]
        unroll_match = pragma["match"]
        pipeline_start, pipeline_end = _line_span(
            source,
            pipeline_match.start(),
            pipeline_match.end(),
        )
        unroll_start, _ = _line_span(
            source,
            unroll_match.start(),
            unroll_match.end(),
        )
        indent = source[unroll_start : unroll_match.start()]
        pipeline_text = pipeline_match.group(0).strip()

        edits.append((pipeline_start, pipeline_end, ""))
        edits.append(
            (
                unroll_start,
                unroll_start,
                f"{indent}{pipeline_text}\n",
            )
        )
        used_pipelines.add(pipeline_match.start())

    updated = source
    for start, end, replacement in sorted(
        edits,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    ):
        updated = updated[:start] + replacement + updated[end:]

    return updated, len(used_pipelines)
