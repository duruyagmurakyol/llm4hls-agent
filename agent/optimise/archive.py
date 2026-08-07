"""Durable best-candidate selection and archive materialisation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from agent.optimise.pareto_frontier import (
    OBJECTIVES,
    annotate_pareto_frontier,
    record_dominates,
)
from agent.optimise.selection import (
    configured_ranking,
    deterministic_selection_key,
    is_fully_verified,
)
from agent.track_a_selection import (
    OFFICIAL_TRACK_A_MODE,
    official_selection_policy,
    select_official_track_a,
    selection_mode,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_config(config_source: Any) -> dict[str, Any]:
    resolved = config_source.resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _objective_values(record: dict[str, Any]) -> tuple[float, ...] | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    values: list[float] = []
    for key in OBJECTIVES:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(float(value))
    return tuple(values)


def _apply_pre_cosim_pareto_gate(
    output_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Skip expensive co-simulation when synthesis proves a candidate cannot win.

    This gate is deliberately conservative. It acts only on candidates already
    marked ``awaiting_cosim`` after CSim, synthesis, timing and resource checks.
    Co-simulation is skipped only when a fully verified current Pareto member is
    no worse in every objective, or when all objective metrics are identical.
    Any genuine performance/resource improvement or trade-off still proceeds to
    required C/RTL co-simulation before it can become selectable.
    """

    frontier = [
        item
        for item in summary.get("pareto_archive", [])
        if isinstance(item, dict) and item.get("fully_verified") is True
    ]
    candidates = [
        item
        for item in summary.get("candidates", [])
        if isinstance(item, dict)
    ]

    for record in candidates:
        if record.get("verdict") != "awaiting_cosim":
            continue

        candidate_values = _objective_values(record)
        if candidate_values is None:
            continue

        identical = next(
            (
                item
                for item in frontier
                if _objective_values(item) == candidate_values
            ),
            None,
        )
        dominators = [
            item
            for item in frontier
            if record_dominates(item, record)
        ]
        dominator = (
            max(
                dominators,
                key=lambda item: int(item.get("candidate_index", 0)),
            )
            if dominators
            else None
        )

        if identical is None and dominator is None:
            continue

        comparator = identical if identical is not None else dominator
        comparator_index = int(comparator.get("candidate_index", 0))
        comparator_label = (
            "verified baseline"
            if comparator_index == 0
            else f"verified candidate_{comparator_index:03d}"
        )
        verdict = (
            "reject_no_change_pre_cosim"
            if identical is not None
            else "reject_dominated_pre_cosim"
        )
        reason = (
            f"Synthesis objectives are identical to the {comparator_label}; "
            "required co-simulation was skipped because the candidate cannot "
            "improve the Pareto archive."
            if identical is not None
            else (
                f"Synthesis objectives are dominated by the {comparator_label}; "
                "required co-simulation was skipped because the candidate cannot "
                "enter the Pareto archive."
            )
        )
        record.update(
            {
                "verdict": verdict,
                "reason": reason,
                "fully_verified": False,
                "cosim": None,
                "cosim_run": False,
                "cosim_skipped": True,
                "cosim_skip_reason": "provisional_pareto_rejection",
                "dominated_by": comparator_index,
            }
        )

        index = int(record.get("candidate_index", 0))
        decision = {
            "schema_version": 1,
            "candidate_index": index,
            "decision": "skip_cosim",
            "reason": "provisional_pareto_rejection",
            "verdict": verdict,
            "dominated_by": comparator_index,
            "objectives": list(OBJECTIVES),
            "candidate_metrics": {
                key: record.get("metrics", {}).get(key) for key in OBJECTIVES
            },
            "comparator_metrics": {
                key: comparator.get("metrics", {}).get(key) for key in OBJECTIVES
            },
        }
        (output_dir / f"candidate_{index:03d}_cosim_decision.json").write_text(
            json.dumps(decision, indent=2) + "\n",
            encoding="utf-8",
        )

    summary["candidates"] = candidates
    return summary


def _is_frequency_compliant(record: dict[str, Any]) -> bool:
    direct = record.get("meets_frequency_requirement")
    if isinstance(direct, bool):
        return direct

    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    nested = metrics.get("meets_minimum_frequency")
    if isinstance(nested, bool):
        return nested

    minimum = metrics.get("minimum_frequency_mhz")
    frequency = metrics.get("frequency_mhz")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        return bool(
            isinstance(frequency, (int, float))
            and not isinstance(frequency, bool)
            and frequency >= minimum
        )

    return True


def _meets_resource_limits(record: dict[str, Any]) -> bool:
    direct = record.get("meets_resource_limits")
    if isinstance(direct, bool):
        return direct
    compliance = record.get("resource_limit_compliance")
    if isinstance(compliance, dict) and isinstance(compliance.get("passed"), bool):
        return bool(compliance["passed"])
    return True


def _source_record(record: dict[str, Any], archived: Path, role: str) -> dict[str, Any]:
    source = _resolve(str(record["candidate_file"]))
    baseline = record.get("candidate_index") == 0

    def validation_value(key: str) -> Any:
        value = record.get(key)
        if baseline and value is None and key != "cosim":
            return True
        return value

    track_a_selection = (
        dict(record.get("track_a_selection") or {})
        if isinstance(record.get("track_a_selection"), dict)
        else {}
    )
    return {
        "role": role,
        "candidate_index": record.get("candidate_index"),
        "candidate_file": _display(source),
        "archived_file": _display(archived),
        "candidate_hash": _hash(source),
        "metrics": dict(record.get("metrics") or {}),
        "fully_verified": is_fully_verified(record),
        "meets_frequency_requirement": _is_frequency_compliant(record),
        "meets_resource_limits": _meets_resource_limits(record),
        "resource_limit_compliance": dict(record.get("resource_limit_compliance") or {}),
        "cost": dict(record.get("cost") or {}),
        "verdict": record.get("verdict"),
        "track_a_selection": track_a_selection,
        "public_score_estimate": track_a_selection.get("public_score_estimate"),
        "official_latency_cycles": track_a_selection.get("official_latency_cycles"),
        "official_validation_credits": track_a_selection.get(
            "official_validation_credits"
        ),
        "validation": {
            "static_validation": validation_value("static_validation"),
            "csim": validation_value("csim"),
            "synthesis": validation_value("synthesis"),
            "cosim": validation_value("cosim"),
        },
    }


def _copy_record(
    record: dict[str, Any] | None,
    destination_stem: Path,
    role: str,
) -> dict[str, Any] | None:
    if record is None:
        return None
    source = _resolve(str(record["candidate_file"]))
    if not source.is_file():
        raise FileNotFoundError(f"Candidate source not found: {source}")
    destination = destination_stem.with_suffix(source.suffix or ".cpp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _hash(destination) != _hash(source):
        raise RuntimeError(f"Archived {role} hash does not match its source")
    return _source_record(record, destination, role)


def preserve_candidate_state(
    config_source: Any,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Persist best-so-far state and return an enriched experiment summary."""
    config = _load_config(config_source)
    selection = dict(config.get("selection") or {})
    mode = selection_mode(selection)
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = annotate_pareto_frontier(output_dir, summary)
    if mode != OFFICIAL_TRACK_A_MODE:
        summary = _apply_pre_cosim_pareto_gate(output_dir, summary)
    archive_dir = output_dir / "candidate_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    baseline_from_summary = summary.get("baseline_record")
    if isinstance(baseline_from_summary, dict):
        baseline = dict(baseline_from_summary)
    else:
        frequency_requirement = (
            summary.get("frequency_requirement")
            if isinstance(summary.get("frequency_requirement"), dict)
            else {}
        )
        verification = config.get("baseline", {}).get("verification")
        verification = dict(verification) if isinstance(verification, dict) else {}
        requires_cosim = bool(config.get("requires_cosim", True))
        baseline_csim = verification.get("csim_passed", True)
        baseline_synthesis = verification.get("synthesis_passed", True)
        baseline_cosim = verification.get("cosim_passed")
        baseline = {
            "candidate_index": 0,
            "candidate_file": config["baseline"]["source"],
            "metrics": dict(summary.get("baseline_metrics") or {}),
            "meets_frequency_requirement": frequency_requirement.get(
                "baseline_meets_requirement"
            ),
            "meets_resource_limits": True,
            "resource_limit_compliance": {"configured": False, "passed": True},
            "fully_verified": bool(
                baseline_csim is True
                and baseline_synthesis is True
                and (baseline_cosim is True if requires_cosim else True)
            ),
            "cosim_required": requires_cosim,
            "verdict": "baseline",
            "static_validation": True,
            "csim": baseline_csim,
            "synthesis": baseline_synthesis,
            "cosim": baseline_cosim,
            "cost": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "tool_calls": 0,
                "tool_seconds": 0.0,
            },
        }

    candidates = [
        dict(item)
        for item in summary.get("candidates", [])
        if isinstance(item, dict)
    ]

    if mode == OFFICIAL_TRACK_A_MODE:
        annotated, official_selected = select_official_track_a(
            [baseline, *candidates],
            config=config,
            output_dir=output_dir,
        )
        baseline = annotated[0]
        candidates = annotated[1:]
    else:
        official_selected = None

    latest = max(
        candidates,
        key=lambda item: int(item.get("candidate_index", -1)),
        default=None,
    )

    verified = [item for item in [baseline, *candidates] if is_fully_verified(item)]
    if mode == OFFICIAL_TRACK_A_MODE:
        best_correct = official_selected
    else:
        best_correct = (
            min(verified, key=lambda item: deterministic_selection_key(item, selection))
            if verified
            else None
        )

    by_index = {
        int(item.get("candidate_index", -1)): item
        for item in [baseline, *candidates]
    }
    pareto_records = []
    for item in summary.get("pareto_archive", []):
        if not isinstance(item, dict):
            continue
        record = by_index.get(int(item.get("candidate_index", -1)), item)
        if (
            is_fully_verified(record)
            and _is_frequency_compliant(record)
            and _meets_resource_limits(record)
        ):
            pareto_records.append(record)

    if mode == OFFICIAL_TRACK_A_MODE:
        best_ppa = (
            official_selected
            if isinstance(official_selected, dict)
            and int(official_selected.get("candidate_index", 0)) > 0
            else None
        )
        selected = official_selected
    else:
        best_ppa = (
            min(pareto_records, key=lambda item: deterministic_selection_key(item, selection))
            if pareto_records
            else None
        )
        selected = best_ppa or best_correct

    previous_path = output_dir / "candidate_state.json"
    previous = {}
    if previous_path.is_file():
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}

    original = previous.get("original_baseline")
    original_file = (
        _resolve(original["archived_file"])
        if isinstance(original, dict) and original.get("archived_file")
        else None
    )
    if (
        mode == OFFICIAL_TRACK_A_MODE
        or not original_file
        or not original_file.is_file()
    ):
        original = _copy_record(
            baseline,
            archive_dir / "original_baseline",
            "original_baseline",
        )

    latest_record = _copy_record(
        latest,
        archive_dir / "latest_candidate",
        "latest_candidate",
    )
    best_correct_record = _copy_record(
        best_correct,
        archive_dir / "best_correct_candidate",
        "best_correct_candidate",
    )
    best_ppa_record = _copy_record(
        best_ppa,
        archive_dir / "best_ppa_candidate",
        "best_ppa_candidate",
    )
    selected_record = _copy_record(
        selected,
        archive_dir / "selected_design",
        "selected_design",
    )

    pareto_dir = archive_dir / "pareto"
    shutil.rmtree(pareto_dir, ignore_errors=True)
    pareto_dir.mkdir(parents=True)
    archived_pareto: list[dict[str, Any]] = []
    for record in pareto_records:
        index = int(record.get("candidate_index", 0))
        label = "baseline" if index == 0 else f"candidate_{index:03d}"
        archived = _copy_record(record, pareto_dir / label, "pareto_member")
        if archived is not None:
            archived_pareto.append(archived)

    policy = (
        official_selection_policy()
        if mode == OFFICIAL_TRACK_A_MODE
        else {
            "mode": mode,
            "ranking": list(configured_ranking(selection)),
            "description": (
                "Select only fully verified designs. Prefer frequency- and "
                "resource-compliant designs, then apply the configured "
                "deterministic ranking and candidate index as the final stable "
                "tie-breaker."
            ),
        }
    )
    selected_track_a = (
        selected.get("track_a_selection")
        if isinstance(selected, dict)
        and isinstance(selected.get("track_a_selection"), dict)
        else {}
    )
    state = {
        "schema_version": 4,
        "selection_policy": policy,
        "frequency_requirement": summary.get("frequency_requirement", {}),
        "resource_limits": summary.get("resource_limits", {}),
        "selected_design_fully_verified": (
            is_fully_verified(selected) if isinstance(selected, dict) else False
        ),
        "selected_design_frequency_compliant": (
            _is_frequency_compliant(selected) if isinstance(selected, dict) else False
        ),
        "selected_design_resource_compliant": (
            _meets_resource_limits(selected) if isinstance(selected, dict) else False
        ),
        "selected_public_score_estimate": selected_track_a.get(
            "public_score_estimate"
        ),
        "selected_official_latency_cycles": selected_track_a.get(
            "official_latency_cycles"
        ),
        "selected_official_validation_credits": selected_track_a.get(
            "official_validation_credits"
        ),
        "original_baseline": original,
        "latest_candidate": latest_record,
        "best_correct_candidate": best_correct_record,
        "best_ppa_candidate": best_ppa_record,
        "selected_design": selected_record,
        "pareto_archive": archived_pareto,
    }
    previous_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    enriched = dict(summary)
    enriched.update(
        {
            "selection_mode": mode,
            "selection_policy": policy,
            "candidates": candidates,
            "original_baseline": state["original_baseline"],
            "latest_candidate": state["latest_candidate"],
            "best_correct_candidate": state["best_correct_candidate"],
            "best_ppa_candidate": state["best_ppa_candidate"],
            "selected_design": state["selected_design"],
            "selected_design_fully_verified": state[
                "selected_design_fully_verified"
            ],
            "selected_design_frequency_compliant": state[
                "selected_design_frequency_compliant"
            ],
            "selected_design_resource_compliant": state[
                "selected_design_resource_compliant"
            ],
            "selected_public_score_estimate": state[
                "selected_public_score_estimate"
            ],
            "selected_official_latency_cycles": state[
                "selected_official_latency_cycles"
            ],
            "selected_official_validation_credits": state[
                "selected_official_validation_credits"
            ],
            "candidate_state": state,
        }
    )
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(enriched, indent=2) + "\n",
        encoding="utf-8",
    )
    return enriched
