"""Deterministic C4 exploitation and C5 recovery/fallback policy.

This module owns the measured-result decisions for the final two slots of the
five-candidate structured search. It does not call a model or validation tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from agent.optimise.selection import deterministic_selection_key
from agent.optimise.structured_exploration import BASELINE_PROMPT_TEMPLATE

REPO_ROOT = Path(__file__).resolve().parents[2]

STRUCTURED_EXPLOIT_REASON = "structured_focused_exploitation"
STRUCTURED_EXPLOIT_FALLBACK_REASON = "structured_exploitation_baseline_fallback"
STRUCTURED_RECOVERY_FALLBACK_REASON = "structured_recovery_baseline_fallback"
STRUCTURED_RECOVERY_REASONS = {
    "pending_latency_recovery_strategy",
    "resource_limit_recovery_from_feasible_pareto",
    "resource_limit_recovery_from_feasible_verified",
    "resource_frequency_balance_from_feasible_parent",
}

FALLBACK_GUIDANCE: dict[int, dict[str, Any]] = {
    4: {
        "name": "loop_schedule_restructuring",
        "objective": (
            "Try one independent loop-schedule transformation from the verified "
            "baseline after the three primary exploration families produced no "
            "refinement-worthy parent."
        ),
        "required_changes": [
            "Choose exactly one legal loop interchange, tiling, fusion, fission, or equivalent schedule restructuring supported by the dependence structure.",
            "Keep the transformation local to the diagnosed target and preserve all bounds, ordering requirements, and numerical behaviour.",
            "Do not reproduce the critical-path, bounded-unroll, or memory-parallelism attempts already evaluated.",
        ],
        "forbidden_changes": [
            "Do not combine multiple unrelated optimisation families.",
            "Do not completely unroll loops or completely partition top-level interface arrays.",
            "Do not apply a loop transformation when dependences make it unsafe.",
        ],
    },
    5: {
        "name": "pipeline_dataflow_restructuring",
        "objective": (
            "Use the final slot for one independent baseline-rooted pipeline or "
            "dataflow restructuring when no bounded measured recovery is available."
        ),
        "required_changes": [
            "Choose exactly one focused loop-pipeline or producer/consumer dataflow transformation supported by the source structure.",
            "Keep buffering bounded and preserve the exact top-level interface, algorithm, and testbench-observed behaviour.",
            "Use the original verified baseline as the sole implementation parent and avoid all previously evaluated source structures.",
        ],
        "forbidden_changes": [
            "Do not combine DATAFLOW and PIPELINE in a conflicting single-process region.",
            "Do not completely partition top-level interface arrays or introduce unbounded buffering.",
            "Do not repeat a previous candidate with cosmetic changes.",
        ],
    },
}


def _record_index(record: dict[str, Any]) -> int:
    value = record.get("candidate_index")
    return int(value) if isinstance(value, int) else 10**9


def select_structured_exploitation_parent(
    records: Iterable[dict[str, Any]],
    selection: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str] | None:
    """Select the strongest explicitly useful C1-C3 candidate for C4."""

    eligible = [
        record
        for record in records
        if (
            isinstance(record.get("candidate_index"), int)
            and 1 <= int(record["candidate_index"]) <= 3
            and record.get("refinement_eligible") is True
        )
    ]
    if not eligible:
        return None
    selected = min(
        eligible,
        key=lambda record: (
            deterministic_selection_key(record, selection),
            _record_index(record),
        ),
    )
    return selected, STRUCTURED_EXPLOIT_REASON


def select_structured_recovery_parent(
    records: Iterable[dict[str, Any]],
    selection: dict[str, Any] | None,
    legacy_selector: Callable[
        [Iterable[dict[str, Any]], dict[str, Any] | None],
        tuple[dict[str, Any], str] | None,
    ],
) -> tuple[dict[str, Any], str] | None:
    """Accept exactly one recovery decision from the established selector."""

    record_list = list(records)
    selected = legacy_selector(record_list, selection)
    if selected is None:
        return None
    parent, reason = selected
    if reason not in STRUCTURED_RECOVERY_REASONS:
        return None
    return parent, reason


def baseline_fallback_parent(
    config: dict[str, Any],
    *,
    candidate_index: int,
) -> tuple[dict[str, Any], str]:
    """Return the verified baseline parent for an unfilled C4 or C5 slot."""

    if candidate_index not in FALLBACK_GUIDANCE:
        raise ValueError("structured baseline fallback is only valid for C4 or C5")
    reason = (
        STRUCTURED_EXPLOIT_FALLBACK_REASON
        if candidate_index == 4
        else STRUCTURED_RECOVERY_FALLBACK_REASON
    )
    return (
        {
            "candidate_index": 0,
            "candidate_file": config["baseline"]["source"],
            "fully_verified": True,
            "verdict": reason,
            "next_candidate_index": candidate_index,
            "strategy_family": FALLBACK_GUIDANCE[candidate_index]["name"],
        },
        reason,
    )


def _load_config(config_source: Any) -> dict[str, Any]:
    resolved = config_source.resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Optimisation config must contain a JSON object")
    return value


def _output_dir(config: dict[str, Any]) -> Path:
    path = Path(str(config["output_dir"])).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _prior_strategy_names(output_dir: Path, candidate_index: int) -> list[str]:
    names: list[str] = []
    for index in range(1, candidate_index):
        path = output_dir / f"candidate_{index:03d}_strategy.json"
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = value.get("name") if isinstance(value, dict) else None
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def prepare_structured_baseline_fallback_prompt(
    config_source: Any,
    *,
    candidate_index: int,
) -> Path:
    """Prepare one independent baseline-rooted C4/C5 fallback prompt."""

    if candidate_index not in FALLBACK_GUIDANCE:
        raise ValueError("structured baseline fallback is only valid for C4 or C5")

    config = _load_config(config_source)
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / BASELINE_PROMPT_TEMPLATE
    if not template_path.is_file():
        initial = output_dir / "candidate_001_prompt.txt"
        if not initial.is_file():
            raise FileNotFoundError(f"Baseline diagnosis prompt not found: {initial}")
        template_path.write_text(initial.read_text(encoding="utf-8"), encoding="utf-8")

    guidance = FALLBACK_GUIDANCE[candidate_index]
    prior = _prior_strategy_names(output_dir, candidate_index)
    required = "\n".join(f"- {item}" for item in guidance["required_changes"])
    forbidden = "\n".join(f"- {item}" for item in guidance["forbidden_changes"])
    prior_text = ", ".join(prior) if prior else "none recorded"
    phase = "exploit_fallback" if candidate_index == 4 else "recovery_fallback"

    prompt = template_path.read_text(encoding="utf-8").rstrip() + f"""

Structured {phase.replace('_', ' ')} contract:
- Search slot: candidate {candidate_index:03d}.
- Implementation parent: original verified baseline (candidate 000).
- Strategy family: {guidance['name']}.
- Previously attempted strategy families: {prior_text}.
- This is one independent final-family attempt, not a repair or continuation of a failed source.

Strategy objective:
{guidance['objective']}

Required changes:
{required}

Forbidden changes:
{forbidden}

Return the complete modified C++ source only. Do not include explanations or Markdown fences.
"""
    prompt_path = output_dir / f"candidate_{candidate_index:03d}_prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    strategy = {
        "name": guidance["name"],
        "parameters": {},
        "reason": guidance["objective"],
        "required_changes": list(guidance["required_changes"]),
        "forbidden_changes": list(guidance["forbidden_changes"]),
        "source_candidate_index": 0,
        "next_candidate_index": candidate_index,
        "trigger": f"structured_{phase}",
        "phase": phase,
        "schedule_slot": candidate_index,
        "compliance_mode": "advisory",
        "compliance_reason": "model_guided_structural_family",
    }
    (output_dir / f"candidate_{candidate_index:03d}_strategy.json").write_text(
        json.dumps(strategy, indent=2) + "\n",
        encoding="utf-8",
    )
    feedback = {
        "previous_candidate_index": 0,
        "next_candidate_index": candidate_index,
        "selected_parent": "verified_baseline",
        "strategy_family": guidance["name"],
        "phase": phase,
        "structured_schedule": True,
        "previous_strategy_families": prior,
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
        "template_file": str(template_path.relative_to(REPO_ROOT)),
    }
    (output_dir / f"candidate_{candidate_index:03d}_feedback.json").write_text(
        json.dumps(feedback, indent=2) + "\n",
        encoding="utf-8",
    )
    return prompt_path


def write_structured_search_decision(
    config_source: Any,
    *,
    candidate_index: int,
    phase: str,
    parent: dict[str, Any],
    reason: str,
) -> Path:
    """Persist the measured decision that prepared C4 or C5."""

    config = _load_config(config_source)
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"candidate_{candidate_index:03d}_search_decision.json"
    path.write_text(
        json.dumps(
            {
                "candidate_index": candidate_index,
                "phase": phase,
                "selected_parent_index": parent.get("candidate_index"),
                "selected_parent_file": parent.get("candidate_file"),
                "parent_reason": reason,
                "structured_schedule": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
