#!/usr/bin/env python3

"""Map hierarchical HLS targets to source regions and infer likely bottleneck causes.

The analyser separates report-proven observations from source-derived hypotheses.
It deliberately uses lightweight, benchmark-independent heuristics rather than
pretending to replace Vitis scheduling analysis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}


def _normalise_target(function: str) -> list[str]:
    lowered = function.lower()
    tokens = [lowered]
    for marker in ("_pipeline_", "pipeline_", "_proc", "_loop"):
        if marker in lowered:
            tokens.extend(part for part in lowered.split(marker) if part)
    tokens.extend(re.findall(r"[a-z][a-z0-9_]{2,}", lowered))
    ignored = {"kernel", "pipeline", "loop", "proc", "top", "function"}
    return sorted({token for token in tokens if token not in ignored}, key=len, reverse=True)


def _source_files(root: Path) -> list[Path]:
    """Return source candidates, including Vitis preprocessed recovery files.

    Generated wrapper/mapper files are retained as low-priority candidates, while
    ``*.pp.*.cpp`` files receive a preference later because they contain the source
    body that Vitis actually synthesised. A caller may explicitly point at a hidden
    ``.autopilot/db`` directory, so solution directories must not be excluded.
    """

    excluded = {".git", "node_modules", "results"}
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part in excluded for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _score_file(path: Path, text: str, tokens: list[str]) -> int:
    haystack = f"{path.name.lower()}\n{text.lower()}"
    score = 0
    for token in tokens:
        if token in path.name.lower():
            score += 8
        score += min(5, haystack.count(token))
    if "#pragma hls" in haystack:
        score += 2
    if re.search(r"\.pp\.\d+\.cpp$", path.name, re.I):
        score += 12
    if path.name.startswith(("apatb_", "mapper_", "hls_design_meta")):
        score -= 12
    return score


def _extract_region(text: str, tokens: list[str], context: int = 18) -> tuple[int, int, str]:
    lines = text.splitlines()
    best_line = 0
    best_score = -1
    for index, line in enumerate(lines):
        lowered = line.lower()
        score = sum(5 for token in tokens if token in lowered)
        if re.search(r"\bfor\s*\(", line):
            score += 1
        if "#pragma hls" in lowered:
            score += 1
        if score > best_score:
            best_line, best_score = index, score
    start = max(0, best_line - context)
    end = min(len(lines), best_line + context + 1)
    numbered = "\n".join(f"{i + 1:5d}: {lines[i]}" for i in range(start, end))
    return start + 1, end, numbered


def _recurrence_updates(code: str) -> list[tuple[str, str, str]]:
    """Detect compound and explicit self-referential recurrence assignments."""

    updates = re.findall(r"\b([a-zA-Z_]\w*)\s*(\+=|-=|\*=)\s*([^;]+);", code)
    seen = {(var, op, expr.strip()) for var, op, expr in updates}

    explicit_pattern = re.compile(
        r"\b([a-zA-Z_]\w*)\s*=\s*\1\s*([+\-*])\s*([^;]+);"
    )
    for match in explicit_pattern.finditer(code):
        var, operator, expr = match.groups()
        item = (var, f"={operator}", expr.strip())
        if item not in seen:
            updates.append(item)
            seen.add(item)
    return updates


def _analyse_region(region: str, report_target: dict[str, Any]) -> dict[str, Any]:
    code = "\n".join(line.split(": ", 1)[-1] for line in region.splitlines())
    max_ii = report_target.get("max_achieved_ii")
    report_category = (report_target.get("primary_diagnosis") or {}).get("category")

    reductions = _recurrence_updates(code)
    array_reads = re.findall(r"\b([a-zA-Z_]\w*)\s*\[[^\]]+\]", code)
    unique_arrays = sorted(set(array_reads))
    has_mul_add = bool(re.search(r"\*[^;\n]+(?:\+|\+=)|(?:\+|\+=)[^;\n]+\*", code))
    pipeline_pragmas = re.findall(r"#\s*pragma\s+HLS\s+PIPELINE[^\n]*", code, re.I)
    unroll_pragmas = re.findall(r"#\s*pragma\s+HLS\s+UNROLL[^\n]*", code, re.I)

    hypotheses: list[dict[str, Any]] = []
    if reductions and max_ii and float(max_ii) > 1:
        variables = sorted({item[0] for item in reductions})
        hypotheses.append(
            {
                "cause": "loop_carried_reduction_recurrence",
                "confidence": 0.88,
                "evidence": [
                    f"achieved_ii={max_ii}",
                    f"source contains recurrence update(s) to {','.join(variables)}",
                ],
                "next_checks": [
                    "inspect Vitis scheduling/dependency messages for recurrence distance and operator latency",
                    "confirm whether floating-point reassociation is allowed",
                ],
                "candidate_transformations": [
                    "partial accumulators or tree reduction",
                    "controlled unrolling matched to independent accumulators",
                ],
            }
        )
    if len(array_reads) >= 3 and max_ii and float(max_ii) > 1:
        hypotheses.append(
            {
                "cause": "memory_access_or_port_pressure",
                "confidence": 0.58,
                "evidence": [
                    f"achieved_ii={max_ii}",
                    f"source region contains {len(array_reads)} array references across {len(unique_arrays)} arrays",
                ],
                "next_checks": [
                    "inspect Vitis warnings for limited memory ports or bus access lower bounds",
                    "identify which arrays are read or written multiple times per iteration",
                ],
                "candidate_transformations": [
                    "array partitioning or local buffering when interfaces permit",
                    "access-pattern restructuring",
                ],
            }
        )
    if report_category == "critical_path" and has_mul_add:
        hypotheses.append(
            {
                "cause": "arithmetic_operator_chain",
                "confidence": 0.7,
                "evidence": [
                    "report shows a critical-path violation",
                    "source region contains multiply/add arithmetic",
                ],
                "next_checks": [
                    "inspect schedule viewer or verbose synthesis log for the critical operator chain",
                ],
                "candidate_transformations": [
                    "balanced reduction tree",
                    "introduce a pipeline boundary without worsening recurrence II",
                ],
            }
        )

    hypotheses.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "source_features": {
            "reduction_updates": [
                {"variable": var, "operator": op, "expression": expr.strip()}
                for var, op, expr in reductions
            ],
            "array_references": array_reads,
            "unique_arrays": unique_arrays,
            "pipeline_pragmas": pipeline_pragmas,
            "unroll_pragmas": unroll_pragmas,
            "multiply_add_pattern": has_mul_add,
        },
        "cause_hypotheses": hypotheses,
        "primary_hypothesis": hypotheses[0] if hypotheses else {
            "cause": "undetermined_from_source",
            "confidence": 0.25,
            "evidence": ["the selected source region does not expose a decisive structural cause"],
            "next_checks": ["collect verbose scheduling and dependency diagnostics"],
            "candidate_transformations": [],
        },
    }


def analyse_source_cause(
    project_root: Path,
    hierarchy: dict[str, Any],
) -> dict[str, Any]:
    target = hierarchy["primary_target"]
    function = str(target["function"])
    tokens = _normalise_target(function)
    candidates: list[tuple[int, Path, str]] = []
    for path in _source_files(project_root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        score = _score_file(path, text, tokens)
        if score > 0:
            candidates.append((score, path, text))
    candidates.sort(key=lambda item: (item[0], -len(str(item[1]))), reverse=True)

    if not candidates:
        return {
            "schema_version": 1,
            "target": function,
            "source_match": None,
            "primary_hypothesis": {
                "cause": "source_not_found",
                "confidence": 0.0,
                "evidence": ["no matching C/C++ source file found under project root"],
                "next_checks": ["supply the benchmark source directory explicitly"],
                "candidate_transformations": [],
            },
            "cause_hypotheses": [],
        }

    score, path, text = candidates[0]
    start, end, region = _extract_region(text, tokens)
    analysis = _analyse_region(region, target)
    return {
        "schema_version": 1,
        "target": function,
        "report_observations": {
            "primary_category": target["primary_diagnosis"]["category"],
            "max_achieved_ii": target.get("max_achieved_ii"),
            "latency_cycles": target.get("latency_cycles"),
            "interval_cycles": target.get("interval_cycles"),
            "diagnostic_evidence": target["primary_diagnosis"].get("evidence", []),
        },
        "source_match": {
            "path": str(path.resolve()),
            "match_score": score,
            "line_start": start,
            "line_end": end,
            "region": region,
        },
        **analysis,
        "protected_regions": hierarchy.get("protected_regions", []),
    }
