"""Deterministic eligibility, constraint and ranking policy for PPA candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESOURCE_METRICS = (
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
)
RESOURCE_ALIASES = {
    "lut": "resources_lut_used",
    "luts": "resources_lut_used",
    "resources_lut_used": "resources_lut_used",
    "ff": "resources_ff_used",
    "ffs": "resources_ff_used",
    "registers": "resources_ff_used",
    "resources_ff_used": "resources_ff_used",
    "dsp": "resources_dsp_used",
    "dsps": "resources_dsp_used",
    "resources_dsp_used": "resources_dsp_used",
    "bram": "resources_bram_used",
    "brams": "resources_bram_used",
    "bram_18k": "resources_bram_used",
    "resources_bram_used": "resources_bram_used",
}
DEFAULT_RESOURCE_WEIGHTS = {
    "resources_lut_used": 1.0,
    "resources_ff_used": 1.0,
    "resources_dsp_used": 100.0,
    "resources_bram_used": 200.0,
}
MANDATORY_RANKING_PREFIX = (
    "fully_verified",
    "frequency",
    "resource_limits",
)
DEFAULT_RANKING = (
    *MANDATORY_RANKING_PREFIX,
    "latency_ns",
    "resource_cost",
    "throughput_period_ns",
    "total_tokens",
    "tool_calls",
    "tool_seconds",
    "candidate_index",
)
ALLOWED_RANKING_FIELDS = set(DEFAULT_RANKING)


def _number(value: Any, *, default: float = float("inf")) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def normalise_resource_limits(raw: Any) -> dict[str, float]:
    """Return task resource limits using synthesis-report metric names."""
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("resource_limits must be an object")

    limits: dict[str, float] = {}
    for original_key, value in raw.items():
        key = RESOURCE_ALIASES.get(str(original_key).strip().lower())
        if key is None:
            raise ValueError(f"Unsupported resource limit: {original_key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Resource limit {original_key} must be a non-negative number")
        limits[key] = float(value)
    return limits


def evaluate_resource_limits(
    metrics: dict[str, Any],
    resource_limits: Any,
) -> dict[str, Any]:
    """Evaluate one metrics record against task-specific resource ceilings."""
    limits = normalise_resource_limits(resource_limits)
    usage = {key: metrics.get(key) for key in limits}
    violations: list[dict[str, Any]] = []
    for key, limit in limits.items():
        actual = metrics.get(key)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            violations.append(
                {
                    "metric": key,
                    "limit": limit,
                    "actual": actual,
                    "reason": "metric_unavailable",
                }
            )
        elif float(actual) > limit:
            violations.append(
                {
                    "metric": key,
                    "limit": limit,
                    "actual": float(actual),
                    "excess": float(actual) - limit,
                    "reason": "limit_exceeded",
                }
            )
    return {
        "configured": bool(limits),
        "passed": not violations,
        "limits": limits,
        "usage": usage,
        "violations": violations,
    }


def is_fully_verified(record: dict[str, Any]) -> bool:
    direct = record.get("fully_verified")
    if isinstance(direct, bool):
        return direct
    requires_cosim = bool(record.get("cosim_required", True))
    return bool(
        record.get("static_validation") is True
        and record.get("csim") is True
        and record.get("synthesis") is True
        and (record.get("cosim") is True if requires_cosim else True)
        and isinstance(record.get("metrics"), dict)
        and record.get("metrics")
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def candidate_cost(output_dir: Path, candidate_index: int) -> dict[str, Any]:
    """Load model and tool cost used to produce and verify one candidate."""
    prefix = f"candidate_{candidate_index:03d}"
    metadata = _load_json(output_dir / f"{prefix}_model_metadata.json")
    reports = [
        _load_json(output_dir / f"{prefix}_csim_validation.json"),
        _load_json(output_dir / f"{prefix}_synthesis.json"),
        _load_json(output_dir / f"{prefix}_cosim.json"),
    ]
    tool_calls = sum(1 for report in reports if report)
    tool_seconds = 0.0
    for report in reports:
        duration = report.get("elapsed_seconds", report.get("duration_seconds"))
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            tool_seconds += float(duration)

    input_tokens = int(metadata.get("input_tokens") or 0)
    output_tokens = int(metadata.get("output_tokens") or 0)
    total_tokens = metadata.get("total_tokens")
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int):
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(total_tokens),
        "tool_calls": tool_calls,
        "tool_seconds": tool_seconds,
    }


def _resource_weights(selection: dict[str, Any]) -> dict[str, float]:
    configured = selection.get("resource_weights", {})
    if not configured:
        return dict(DEFAULT_RESOURCE_WEIGHTS)
    if not isinstance(configured, dict):
        raise ValueError("selection.resource_weights must be an object")
    weights = dict(DEFAULT_RESOURCE_WEIGHTS)
    for original_key, value in configured.items():
        key = RESOURCE_ALIASES.get(str(original_key).strip().lower())
        if key is None:
            raise ValueError(f"Unsupported resource weight: {original_key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Resource weight {original_key} must be non-negative")
        weights[key] = float(value)
    return weights


def resource_cost(record: dict[str, Any], selection: dict[str, Any] | None = None) -> float:
    """Return a deterministic scalar resource cost for final tie-breaking."""
    selection = selection or {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    compliance = (
        record.get("resource_limit_compliance")
        if isinstance(record.get("resource_limit_compliance"), dict)
        else {}
    )
    limits = compliance.get("limits") if isinstance(compliance.get("limits"), dict) else {}
    if limits:
        total = 0.0
        for key, limit in limits.items():
            actual = _number(metrics.get(key))
            if actual == float("inf"):
                return actual
            total += actual / max(float(limit), 1.0)
        return total

    weights = _resource_weights(selection)
    total = 0.0
    for key in RESOURCE_METRICS:
        value = _number(metrics.get(key))
        if value == float("inf"):
            return value
        total += value * weights[key]
    return total


def configured_ranking(selection: dict[str, Any] | None = None) -> tuple[str, ...]:
    selection = selection or {}
    configured = selection.get("ranking")
    if configured is None:
        return DEFAULT_RANKING
    if not isinstance(configured, list) or not configured or not all(
        isinstance(item, str) and item for item in configured
    ):
        raise ValueError("selection.ranking must be a non-empty list of strings")
    unknown = [item for item in configured if item not in ALLOWED_RANKING_FIELDS]
    if unknown:
        raise ValueError("Unsupported selection ranking fields: " + ", ".join(unknown))

    result = list(MANDATORY_RANKING_PREFIX)
    for field in configured:
        if field not in result:
            result.append(field)
    if "candidate_index" not in result:
        result.append("candidate_index")
    return tuple(result)


def deterministic_selection_key(
    record: dict[str, Any],
    selection: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    """Return the configured, deterministic lexicographic ranking key."""
    selection = selection or {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    cost = record.get("cost") if isinstance(record.get("cost"), dict) else {}
    compliance = (
        record.get("resource_limit_compliance")
        if isinstance(record.get("resource_limit_compliance"), dict)
        else {}
    )

    values = {
        "fully_verified": 0.0 if is_fully_verified(record) else 1.0,
        "frequency": 0.0 if record.get("meets_frequency_requirement") is True else 1.0,
        "resource_limits": 0.0 if compliance.get("passed", True) is True else 1.0,
        "latency_ns": _number(metrics.get("latency_ns")),
        "throughput_period_ns": _number(metrics.get("throughput_period_ns")),
        "resource_cost": resource_cost(record, selection),
        "total_tokens": _number(cost.get("total_tokens"), default=0.0),
        "tool_calls": _number(cost.get("tool_calls"), default=0.0),
        "tool_seconds": _number(cost.get("tool_seconds"), default=0.0),
        "candidate_index": _number(record.get("candidate_index")),
    }
    return tuple(values[field] for field in configured_ranking(selection))
