"""Concise, deterministic descriptions of attempted repair strategies."""

from __future__ import annotations

import difflib
import hashlib
import json
from typing import Any, Iterable


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def build_strategy(
    *,
    before_source: str,
    candidate_source: str,
    editable_file: str,
    max_changed_lines: int = 12,
) -> dict[str, Any]:
    """Describe the source edit without asking the model to explain its intent."""
    before_lines = before_source.splitlines()
    candidate_lines = candidate_source.splitlines()
    matcher = difflib.SequenceMatcher(a=before_lines, b=candidate_lines, autojunk=False)

    operations: list[dict[str, Any]] = []
    changed_lines: list[str] = []
    removed_count = 0
    added_count = 0

    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        removed = before_lines[before_start:before_end]
        added = candidate_lines[after_start:after_end]
        removed_count += len(removed)
        added_count += len(added)
        operations.append(
            {
                "kind": tag,
                "before_lines": [before_start + 1, before_end],
                "candidate_lines": [after_start + 1, after_end],
                "removed": removed,
                "added": added,
            }
        )
        changed_lines.extend(f"- {line}" for line in removed)
        changed_lines.extend(f"+ {line}" for line in added)

    accepted_change = bool(operations)
    if accepted_change:
        summary = (
            f"Changed {editable_file}: removed {removed_count} line(s) and added "
            f"{added_count} line(s) across {len(operations)} edit block(s)."
        )
        fingerprint_payload = {
            "editable_file": editable_file,
            "operations": operations,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    else:
        summary = "No editable source change was accepted for this attempt."
        fingerprint = None

    shown = changed_lines[:max_changed_lines]
    omitted = max(0, len(changed_lines) - len(shown))
    if omitted:
        shown.append(f"... {omitted} additional changed line(s) omitted")

    return {
        "schema_version": 1,
        "editable_file": editable_file,
        "summary": summary,
        "accepted_change": accepted_change,
        "removed_line_count": removed_count,
        "added_line_count": added_count,
        "edit_block_count": len(operations),
        "changed_lines": shown,
        "operations": operations,
        "fingerprint": fingerprint,
        "before_hash": _source_hash(before_source),
        "candidate_hash": _source_hash(candidate_source),
    }


def strategy_feedback_evidence(strategy: dict[str, Any]) -> list[str]:
    """Render strategy memory as concise evidence for the next repair attempt."""
    lines = [f"Previous strategy: {strategy.get('summary', 'unknown strategy')}"]
    for changed in strategy.get("changed_lines", []):
        lines.append(f"Previous strategy change: {changed}")
    fingerprint = strategy.get("fingerprint")
    if fingerprint:
        lines.append(f"Previous strategy fingerprint: {fingerprint}")
    return _unique(lines)


DO_NOT_REPEAT_CONSTRAINT = (
    "Do not repeat the previous repair strategy unchanged; make a materially "
    "different source edit that responds to the failure evidence."
)
