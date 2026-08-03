"""Durable best-candidate selection and archive materialisation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTION_OBJECTIVES = (
    "latency_ns",
    "throughput_period_ns",
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
)
LEGACY_OBJECTIVE_FALLBACKS = {
    "latency_ns": "latency_best_cycles",
    "throughput_period_ns": "interval_min_cycles",
}


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


def _number(value: Any) -> float:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else float("inf")
    )


def _objective(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        value = metrics.get(LEGACY_OBJECTIVE_FALLBACKS.get(key, ""))
    return _number(value)


def _selection_key(record: dict[str, Any]) -> tuple[float, ...]:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    index = record.get("candidate_index")
    return tuple(_objective(metrics, key) for key in SELECTION_OBJECTIVES) + (
        float(index) if isinstance(index, int) else float("inf"),
    )


def _is_verified(record: dict[str, Any]) -> bool:
    if record.get("candidate_index") == 0:
        return True
    metrics = record.get("metrics")
    return bool(
        record.get("static_validation") is True
        and record.get("csim") is True
        and record.get("synthesis") is True
        and isinstance(metrics, dict)
        and metrics
    )


def _is_frequency_compliant(record: dict[str, Any]) -> bool:
    direct = record.get("meets_frequency_requirement")
    if isinstance(direct, bool):
        return direct
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    nested = metrics.get("meets_minimum_frequency")
    return nested is True


def _source_record(record: dict[str, Any], archived: Path, role: str) -> dict[str, Any]:
    source = _resolve(str(record["candidate_file"]))
    baseline = record.get("candidate_index") == 0

    def validation_value(key: str) -> Any:
        value = record.get(key)
        return True if baseline and value is None else value

    return {
        "role": role,
        "candidate_index": record.get("candidate_index"),
        "candidate_file": _display(source),
        "archived_file": _display(archived),
        "candidate_hash": _hash(source),
        "metrics": dict(record.get("metrics") or {}),
        "meets_frequency_requirement": _is_frequency_compliant(record),
        "verdict": record.get("verdict"),
        "validation": {
            "static_validation": validation_value("static_validation"),
            "csim": validation_value("csim"),
            "synthesis": validation_value("synthesis"),
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
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir / "candidate_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    frequency_requirement = (
        summary.get("frequency_requirement")
        if isinstance(summary.get("frequency_requirement"), dict)
        else {}
    )
    baseline = {
        "candidate_index": 0,
        "candidate_file": config["baseline"]["source"],
        "metrics": dict(summary.get("baseline_metrics") or {}),
        "meets_frequency_requirement": frequency_requirement.get(
            "baseline_meets_requirement"
        ),
        "verdict": "baseline",
        "static_validation": True,
        "csim": True,
        "synthesis": True,
    }
    candidates = [
        item for item in summary.get("candidates", []) if isinstance(item, dict)
    ]
    latest = max(
        candidates,
        key=lambda item: int(item.get("candidate_index", -1)),
        default=None,
    )
    verified = [baseline, *[item for item in candidates if _is_verified(item)]]
    best_correct = min(verified, key=_selection_key)
    compliant_verified = [item for item in verified if _is_frequency_compliant(item)]
    best_compliant_correct = (
        min(compliant_verified, key=_selection_key) if compliant_verified else None
    )

    pareto_records = [
        item
        for item in summary.get("pareto_archive", [])
        if isinstance(item, dict) and _is_frequency_compliant(item)
    ]
    best_ppa = min(pareto_records, key=_selection_key) if pareto_records else None
    selected = best_ppa or best_compliant_correct or best_correct

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
    if not original_file or not original_file.is_file():
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

    state = {
        "schema_version": 2,
        "selection_policy": (
            "Require the configured minimum frequency, then select from the Pareto "
            "archive using actual latency, throughput period, LUT, FF, DSP and BRAM; "
            "fall back to the best frequency-compliant verified design, then the best "
            "correct design only when no compliant implementation exists."
        ),
        "frequency_requirement": frequency_requirement,
        "selected_design_frequency_compliant": _is_frequency_compliant(selected),
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
            "original_baseline": state["original_baseline"],
            "latest_candidate": state["latest_candidate"],
            "best_correct_candidate": state["best_correct_candidate"],
            "best_ppa_candidate": state["best_ppa_candidate"],
            "selected_design": state["selected_design"],
            "selected_design_frequency_compliant": state[
                "selected_design_frequency_compliant"
            ],
            "candidate_state": state,
        }
    )
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(enriched, indent=2) + "\n",
        encoding="utf-8",
    )
    return enriched
