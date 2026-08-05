"""Lossless prompt compaction for HLS repair and optimisation calls.

The compactor removes prompt-only redundancy while preserving the current
editable/parent source, hard constraints, latest tool evidence, and semantic
contracts. It never modifies files used by Vitis.
"""

from __future__ import annotations

import math
import re
from typing import Any

_PARENT_SOURCE_MARKERS = (
    "Previous candidate source:",
    "Pareto candidate source:",
    "Feasible parent source to modify:",
)
_BASELINE_SOURCE_MARKER = "Original baseline source:"
_UNFENCED_SOURCE_MARKERS = (
    *_PARENT_SOURCE_MARKERS,
    _BASELINE_SOURCE_MARKER,
    "Baseline source:",
)
_METRIC_ALIASES = {
    "clock_period_ns": "clk_ns",
    "frequency_mhz": "freq_mhz",
    "minimum_frequency_mhz": "min_freq_mhz",
    "maximum_clock_period_ns": "max_clk_ns",
    "latency_best_cycles": "lat_cycles",
    "latency_average_cycles": "lat_avg_cycles",
    "latency_worst_cycles": "lat_worst_cycles",
    "latency_ns": "lat_ns",
    "latency_best_ns": "lat_best_ns",
    "latency_average_ns": "lat_avg_ns",
    "latency_worst_ns": "lat_worst_ns",
    "interval_min_cycles": "ii_min",
    "interval_max_cycles": "ii_max",
    "throughput_period_ns": "throughput_ns",
    "resources_lut_used": "lut",
    "resources_ff_used": "ff",
    "resources_dsp_used": "dsp",
    "resources_bram_used": "bram",
}


def _approx_tokens(text: str) -> int:
    """Return a deterministic audit estimate, not a billing-token count."""
    return math.ceil(len(text) / 4)


def _collapse_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _strip_cpp_comments(source: str) -> tuple[str, int]:
    """Remove C/C++ comments without touching strings, chars, or raw strings."""
    output: list[str] = []
    index = 0
    removed = 0
    length = len(source)

    while index < length:
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            if end == -1:
                removed += length - index
                break
            removed += end - index
            output.append("\n")
            index = end + 1
            continue

        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end == -1:
                comment = source[index:]
                output.extend("\n" for character in comment if character == "\n")
                removed += len(comment) - comment.count("\n")
                break
            comment = source[index : end + 2]
            output.extend("\n" for character in comment if character == "\n")
            removed += len(comment) - comment.count("\n")
            index = end + 2
            continue

        if source.startswith('R"', index):
            delimiter_end = source.find("(", index + 2, min(length, index + 20))
            if delimiter_end != -1:
                delimiter = source[index + 2 : delimiter_end]
                terminator = ")" + delimiter + '"'
                raw_end = source.find(terminator, delimiter_end + 1)
                if raw_end != -1:
                    raw_end += len(terminator)
                    output.append(source[index:raw_end])
                    index = raw_end
                    continue

        character = source[index]
        if character in {'"', "'"}:
            quote = character
            start = index
            index += 1
            while index < length:
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    index += 1
                    break
                index += 1
            output.append(source[start:index])
            continue

        output.append(character)
        index += 1

    compacted = _collapse_blank_lines("".join(output))
    return compacted, removed


def _drop_redundant_baseline_source(prompt: str) -> tuple[str, bool]:
    """Drop the original source only when a current parent source is supplied."""
    if not any(marker in prompt for marker in _PARENT_SOURCE_MARKERS):
        return prompt, False
    marker_index = prompt.rfind(_BASELINE_SOURCE_MARKER)
    if marker_index == -1:
        return prompt, False
    # Current prompt builders place the original baseline last. Be conservative
    # and only drop it when there is no later known top-level source section.
    if any(
        prompt.find(marker, marker_index + len(_BASELINE_SOURCE_MARKER)) != -1
        for marker in _UNFENCED_SOURCE_MARKERS
    ):
        return prompt, False
    return prompt[:marker_index].rstrip() + "\n", True


def _compact_unfenced_source_sections(prompt: str) -> tuple[str, int]:
    locations = sorted(
        (prompt.find(marker), marker)
        for marker in _UNFENCED_SOURCE_MARKERS
        if prompt.find(marker) != -1
    )
    if not locations:
        return prompt, 0

    removed = 0
    result = prompt
    # Work backwards so earlier offsets remain valid.
    for position, marker in reversed(locations):
        body_start = position + len(marker)
        if body_start < len(result) and result[body_start] == "\n":
            body_start += 1
        later_positions = [
            result.find(other, body_start)
            for other in _UNFENCED_SOURCE_MARKERS
            if result.find(other, body_start) != -1
        ]
        body_end = min(later_positions) if later_positions else len(result)
        source = result[body_start:body_end]
        compacted, source_removed = _strip_cpp_comments(source)
        removed += source_removed
        result = result[:body_start] + compacted + result[body_end:]
    return result, removed


def _compact_editable_fenced_source(prompt: str) -> tuple[str, int]:
    pattern = re.compile(
        r"(?P<header>^EDITABLE FILE:[^\n]*\n```(?:cpp|c\+\+|c)?\n)"
        r"(?P<body>.*?)"
        r"(?P<footer>\n```)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    removed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal removed
        compacted, source_removed = _strip_cpp_comments(match.group("body"))
        removed += source_removed
        return match.group("header") + compacted.rstrip() + match.group("footer")

    return pattern.sub(replace, prompt), removed


def _compact_metric_blocks(prompt: str) -> tuple[str, int]:
    lines = prompt.splitlines()
    output: list[str] = []
    index = 0
    compacted_blocks = 0

    while index < len(lines):
        line = lines[index]
        output.append(line)
        if not line.strip().lower().endswith("metrics:"):
            index += 1
            continue

        cursor = index + 1
        metrics: list[tuple[str, str]] = []
        while cursor < len(lines):
            match = re.fullmatch(r"\s*-\s*([A-Za-z0-9_]+):\s*(.*?)\s*", lines[cursor])
            if not match:
                break
            key, value = match.groups()
            if value not in {"", "None", "null"}:
                metrics.append((key, value))
            cursor += 1

        if len(metrics) < 3:
            index += 1
            continue

        values = dict(metrics)
        for group in (
            ("latency_best_cycles", "latency_average_cycles", "latency_worst_cycles"),
            ("latency_best_ns", "latency_average_ns", "latency_worst_ns"),
        ):
            best, average, worst = group
            if values.get(best) == values.get(average) == values.get(worst):
                values.pop(average, None)
                values.pop(worst, None)

        if values.get("interval_min_cycles") == values.get("interval_max_cycles"):
            interval = values.pop("interval_min_cycles", None)
            values.pop("interval_max_cycles", None)
            if interval is not None:
                values["ii_cycles"] = interval

        ordered_keys = [key for key, _ in metrics if key in values]
        if "ii_cycles" in values:
            insertion = next(
                (
                    position
                    for position, key in enumerate(ordered_keys)
                    if key.startswith("throughput")
                ),
                len(ordered_keys),
            )
            ordered_keys.insert(insertion, "ii_cycles")
        seen: set[str] = set()
        compacted = []
        for key in ordered_keys:
            if key in seen:
                continue
            seen.add(key)
            compacted.append(f"{_METRIC_ALIASES.get(key, key)}={values[key]}")
        output.append("  " + ", ".join(compacted))
        compacted_blocks += 1
        index = cursor

    return "\n".join(output), compacted_blocks


def _dedupe_instruction_lines(prompt: str) -> tuple[str, int]:
    seen: set[str] = set()
    output: list[str] = []
    removed = 0
    instruction = re.compile(r"^\s*(?:- |\d+\. )")

    for line in prompt.splitlines():
        if not instruction.match(line):
            output.append(line)
            continue
        normalised = re.sub(r"\s+", " ", line.strip()).casefold()
        if normalised in seen:
            removed += 1
            continue
        seen.add(normalised)
        output.append(line)
    return "\n".join(output), removed


def compact_user_prompt(prompt: str) -> tuple[str, dict[str, Any]]:
    """Return a compact prompt and deterministic audit statistics.

    The transformation is intentionally conservative. It keeps the current
    parent/editable source, hard constraints, latest evidence and task context.
    """
    original = prompt.replace("\r\n", "\n")
    compacted, dropped_baseline = _drop_redundant_baseline_source(original)
    compacted, unfenced_comment_chars = _compact_unfenced_source_sections(compacted)
    compacted, fenced_comment_chars = _compact_editable_fenced_source(compacted)
    compacted, metric_blocks = _compact_metric_blocks(compacted)
    compacted, duplicate_lines = _dedupe_instruction_lines(compacted)
    compacted = _collapse_blank_lines(compacted)

    stats: dict[str, Any] = {
        "enabled": True,
        "original_characters": len(original),
        "compacted_characters": len(compacted),
        "characters_saved": len(original) - len(compacted),
        "reduction_percent": (
            100.0 * (len(original) - len(compacted)) / len(original)
            if original
            else 0.0
        ),
        "estimated_original_tokens": _approx_tokens(original),
        "estimated_compacted_tokens": _approx_tokens(compacted),
        "dropped_redundant_baseline_source": dropped_baseline,
        "source_comment_characters_removed": (
            unfenced_comment_chars + fenced_comment_chars
        ),
        "metric_blocks_compacted": metric_blocks,
        "duplicate_instruction_lines_removed": duplicate_lines,
    }
    return compacted, stats
