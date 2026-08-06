"""Competition-native final selection for FPT Track-A tasks.

Research tasks keep the existing Pareto selector. Track-A tasks explicitly using
``official_track_a`` rank publicly verified designs by the organiser-facing
score estimate and worst/average latency cycles, then use cost and deterministic
tie-breakers. The hidden-test gate remains unavailable to the agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from agent.optimise.selection import is_fully_verified, resource_cost
from agent.track_a_scoring import (
    estimate_public_track_a_score,
    official_latency_cycles,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_TRACK_A_MODE = "official_track_a"
RESEARCH_PARETO_MODE = "research_pareto"
OFFICIAL_TRACK_A_RANKING = (
    "public_correctness",
    "required_cosim",
    "synthesis",
    "public_score_estimate",
    "official_latency_cycles",
    "official_validation_credits",
    "total_tokens",
    "resource_cost",
    "candidate_index",
)


def _number(value: Any, *, default: float = float("inf")) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def selection_mode(selection: dict[str, Any] | None) -> str:
    """Return and validate the configured final-selection mode."""

    value = str((selection or {}).get("mode", RESEARCH_PARETO_MODE)).strip()
    if value not in {RESEARCH_PARETO_MODE, OFFICIAL_TRACK_A_MODE}:
        raise ValueError(
            "selection.mode must be research_pareto or official_track_a"
        )
    return value


def _original_latency(output_dir: Path, baseline_metrics: dict[str, Any]) -> Any:
    """Load original-kernel latency without launching another synthesis call."""

    record = _load_json(output_dir / "original_scoring_baseline.json")
    latency = record.get("official_latency_cycles")
    if isinstance(latency, (int, float)) and not isinstance(latency, bool):
        return latency

    digest = record.get("candidate_hash")
    if isinstance(digest, str) and digest:
        direct = output_dir / "synthesis" / digest[:12] / "result.json"
        reports = [direct] if direct.is_file() else sorted(
            (output_dir / "synthesis").glob("*/result.json")
        )
        for path in reports:
            synthesis = _load_json(path)
            if (
                synthesis.get("candidate_hash") == digest
                and synthesis.get("passed") is True
            ):
                latency = official_latency_cycles(
                    dict(synthesis.get("metrics") or {})
                )
                if latency is not None:
                    return latency

    # For initially correct optimisation tasks, the verified safety baseline is
    # the untouched original public kernel. This fallback avoids a redundant
    # synthesis call while preserving the original-vs-candidate comparison.
    if record.get("synthesis_passed") is not False:
        return official_latency_cycles(baseline_metrics)
    return None


def official_validation_credits(record: dict[str, Any]) -> int:
    """Return weighted credits directly attributable to candidate validation."""

    direct = record.get("official_validation_credits")
    if isinstance(direct, int) and not isinstance(direct, bool) and direct >= 0:
        return direct
    if record.get("candidate_index") == 0:
        return 0

    credits = 0
    if record.get("csim") is not None:
        credits += 1
    if record.get("synthesis") is not None:
        credits += 4
    if record.get("cosim") is not None:
        credits += 20
    return credits


def annotate_official_track_a_record(
    record: dict[str, Any],
    *,
    difficulty: int,
    original_latency_cycles: int | float | None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach public score and competition ranking evidence to one design."""

    annotated = dict(record)
    metrics = dict(record.get("metrics") or {})
    requires_cosim = bool(record.get("cosim_required", False))
    public_correct = bool(
        record.get("csim") is True
        and (not requires_cosim or record.get("cosim") is True)
    )
    synthesis_passed = record.get("synthesis") is True
    latency = official_latency_cycles(metrics)
    score = estimate_public_track_a_score(
        difficulty=difficulty,
        public_correct=public_correct,
        synthesis_passed=synthesis_passed,
        original_latency_cycles=original_latency_cycles,
        candidate_latency_cycles=latency,
    )
    cost = record.get("cost") if isinstance(record.get("cost"), dict) else {}
    evidence = {
        "public_correct": public_correct,
        "required_cosim": requires_cosim,
        "required_cosim_passed": (
            record.get("cosim") is True if requires_cosim else None
        ),
        "synthesis_passed": synthesis_passed,
        "public_score_estimate": score["public_score_estimate"],
        "maximum_score": score["maximum_score"],
        "acceleration": score["acceleration"],
        "ppa_norm": score["ppa_norm"],
        "official_latency_cycles": latency,
        "official_validation_credits": official_validation_credits(record),
        "total_tokens": int(cost.get("total_tokens") or 0),
        "resource_cost": resource_cost(record, selection),
    }
    annotated["track_a_selection"] = evidence
    annotated["public_score_estimate"] = evidence["public_score_estimate"]
    annotated["official_latency_cycles"] = latency
    annotated["official_validation_credits"] = evidence[
        "official_validation_credits"
    ]
    return annotated


def official_track_a_selection_key(
    record: dict[str, Any],
) -> tuple[float, ...]:
    """Return the exact deterministic Track-A final-selection key."""

    evidence = (
        record.get("track_a_selection")
        if isinstance(record.get("track_a_selection"), dict)
        else {}
    )
    requires_cosim = evidence.get("required_cosim") is True
    required_cosim_passed = (
        evidence.get("required_cosim_passed") is True if requires_cosim else True
    )
    return (
        0.0 if evidence.get("public_correct") is True else 1.0,
        0.0 if required_cosim_passed else 1.0,
        0.0 if evidence.get("synthesis_passed") is True else 1.0,
        -_number(evidence.get("public_score_estimate"), default=0.0),
        _number(evidence.get("official_latency_cycles")),
        _number(evidence.get("official_validation_credits"), default=0.0),
        _number(evidence.get("total_tokens"), default=0.0),
        _number(evidence.get("resource_cost")),
        _number(record.get("candidate_index")),
    )


def select_official_track_a(
    records: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Annotate and select the highest-ranked public Track-A design."""

    values = [dict(record) for record in records]
    if not values:
        return [], None

    track_a = config.get("track_a") if isinstance(config.get("track_a"), dict) else {}
    difficulty = int(track_a.get("difficulty", 1))
    selection = config.get("selection") if isinstance(config.get("selection"), dict) else {}
    baseline_metrics = dict(values[0].get("metrics") or {})
    original_latency = _original_latency(output_dir, baseline_metrics)
    annotated = [
        annotate_official_track_a_record(
            record,
            difficulty=difficulty,
            original_latency_cycles=original_latency,
            selection=selection,
        )
        for record in values
    ]
    eligible = [
        record
        for record in annotated
        if is_fully_verified(record)
        and record["track_a_selection"]["public_correct"] is True
        and record["track_a_selection"]["synthesis_passed"] is True
    ]
    selected = min(eligible, key=official_track_a_selection_key) if eligible else None
    return annotated, selected


def official_selection_policy() -> dict[str, Any]:
    return {
        "mode": OFFICIAL_TRACK_A_MODE,
        "ranking": list(OFFICIAL_TRACK_A_RANKING),
        "description": (
            "Select only publicly correct, required-co-simulation-compliant, "
            "synthesised designs. Maximise the public Track-A score estimate; "
            "then minimise official latency cycles, weighted validation credits, "
            "tokens, resource cost and candidate index. The 5 ns target is "
            "reported separately rather than silently treated as a hidden gate."
        ),
    }
