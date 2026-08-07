"""Keep structured exploration moving after an early rejected candidate.

The bounded runner normally recognises completed C1-C3 slots from their audit
metadata. A candidate can nevertheless be rejected before CSim and marked
``refinement_eligible = false``. The legacy selector then requests a generic
baseline restart, whose prompt builder is intended for measured no-gain results
rather than an early static rejection.

This compatibility guard sits underneath the bounded runner. It recognises a
completed structured exploration slot from persisted strategy, feedback, or
prompt evidence and returns the verified baseline for the next exploration
slot. Diagnosis-aware runs read the immutable per-run exploration plan; older
runs without that file retain the historical fixed family mapping.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent.optimise import runner_legacy

STRUCTURED_PARENT_REASON = "structured_baseline_exploration"
DEFAULT_EXPLORATION_FAMILIES = {
    1: "critical_path_restructuring",
    2: "bounded_unroll",
    3: "memory_parallelism",
}
EXPLORATION_PLAN_FILE = "structured_exploration_plan.json"

_ORIGINAL_SELECT: Any = None


def _load_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _candidate_output_dir(record: dict[str, Any]) -> Path | None:
    value = record.get("candidate_file")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(runner_legacy.REPO_ROOT) / path
    return path.parent


def _exploration_family(output_dir: Path, index: int) -> str:
    plan = _load_optional(output_dir / EXPLORATION_PLAN_FILE)
    selected = plan.get("selected_strategy_families")
    if (
        isinstance(selected, list)
        and len(selected) == 3
        and all(isinstance(item, str) and item for item in selected)
        and index in {1, 2, 3}
    ):
        return selected[index - 1]
    return DEFAULT_EXPLORATION_FAMILIES[index]


def _strategy_evidence(output_dir: Path, index: int, family: str) -> bool:
    strategy = _load_optional(
        output_dir / f"candidate_{index:03d}_strategy.json"
    )
    return bool(
        strategy.get("name") == family
        and strategy.get("source_candidate_index") == 0
        and (
            strategy.get("phase") == "explore"
            or strategy.get("trigger") == STRUCTURED_PARENT_REASON
            or strategy.get("compliance_mode") == "advisory"
        )
    )


def _feedback_evidence(output_dir: Path, index: int, family: str) -> bool:
    feedback = _load_optional(
        output_dir / f"candidate_{index:03d}_feedback.json"
    )
    return bool(
        feedback.get("previous_candidate_index") == 0
        and feedback.get("strategy_family") == family
        and (
            feedback.get("structured_schedule") is True
            or feedback.get("phase") == "explore"
        )
    )


def _prompt_evidence(output_dir: Path, index: int, family: str) -> bool:
    for suffix in ("effective_prompt.txt", "prompt.txt"):
        path = output_dir / f"candidate_{index:03d}_{suffix}"
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if (
            "Structured exploration contract:" in text
            and f"Strategy family: {family}." in text
            and "Implementation parent: original verified baseline" in text
        ):
            return True
    return False


def _slot_has_structured_evidence(
    output_dir: Path,
    index: int,
) -> bool:
    family = _exploration_family(output_dir, index)
    return any(
        (
            _strategy_evidence(output_dir, index, family),
            _feedback_evidence(output_dir, index, family),
            _prompt_evidence(output_dir, index, family),
        )
    )


def _structured_baseline_transition(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], str] | None:
    indexed = {
        int(record["candidate_index"]): record
        for record in records
        if isinstance(record.get("candidate_index"), int)
    }
    if not indexed:
        return None

    latest_index = max(indexed)
    next_index = latest_index + 1
    if next_index not in {2, 3}:
        return None
    if set(range(1, next_index)) - set(indexed):
        return None

    output_dirs = {
        output_dir
        for index in range(1, next_index)
        if (output_dir := _candidate_output_dir(indexed[index])) is not None
    }
    if len(output_dirs) != 1:
        return None
    output_dir = next(iter(output_dirs))

    if not all(
        _slot_has_structured_evidence(output_dir, index)
        for index in range(1, next_index)
    ):
        return None

    return (
        {
            "candidate_index": 0,
            "candidate_file": "verified_baseline",
            "fully_verified": True,
            "verdict": STRUCTURED_PARENT_REASON,
            "next_candidate_index": next_index,
            "strategy_family": _exploration_family(output_dir, next_index),
        },
        STRUCTURED_PARENT_REASON,
    )


def _guarded_select(
    records: Iterable[dict[str, Any]],
    selection: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str] | None:
    record_list = list(records)
    transition = _structured_baseline_transition(record_list)
    if transition is not None:
        return transition
    return _ORIGINAL_SELECT(record_list, selection)


def install_structured_transition_runtime() -> None:
    """Install the idempotent C1->C2->C3 transition compatibility guard."""

    global _ORIGINAL_SELECT

    marker = "_structured_transition_runtime_installed"
    if getattr(runner_legacy, marker, False):
        return

    _ORIGINAL_SELECT = runner_legacy.select_refinement_parent
    runner_legacy.select_refinement_parent = _guarded_select
    setattr(runner_legacy, marker, True)


install_structured_transition_runtime()
