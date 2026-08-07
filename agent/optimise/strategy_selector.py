"""Conservative diagnosis-aware selection of generic HLS exploration families.

The selector is intentionally evidence gated.  If baseline artefacts are absent,
ambiguous, or do not match one of the supported generic patterns, it returns the
historical fixed exploration tuple unchanged.  This keeps existing working
behaviour as the safe fallback while allowing stronger architecture-level
search when the measured diagnosis is actionable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.optimise.search_policy import (
    DEFAULT_EXPLORATION_STRATEGY_FAMILIES,
    LAYER_ONE_STRATEGY_FAMILIES,
)

PLAN_FILE = "structured_exploration_plan.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _diagnosis_categories(hierarchy: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    primary = hierarchy.get("primary_target")
    ranked = hierarchy.get("ranked_targets")
    records = []
    if isinstance(primary, dict):
        records.append(primary)
    if isinstance(ranked, list):
        records.extend(item for item in ranked if isinstance(item, dict))

    for record in records:
        diagnosis = record.get("primary_diagnosis")
        if not isinstance(diagnosis, dict):
            continue
        category = diagnosis.get("category")
        if isinstance(category, str):
            categories.append(category.casefold())
    return categories


def _source_text(output_dir: Path, target: dict[str, Any]) -> str:
    candidates = [
        output_dir / "active_baseline.cpp",
        output_dir / "verified_baseline.cpp",
    ]
    for path in candidates:
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                pass

    excerpt = target.get("source_excerpt")
    return excerpt if isinstance(excerpt, str) else ""


def _looks_like_sliding_window(source: str) -> bool:
    """Recognise explicit neighbouring-index reuse without benchmark names."""

    if not source:
        return False
    analysed = re.sub(r"//.*?$|/\*.*?\*/", "", source, flags=re.M | re.S)
    neighbour_accesses = re.findall(
        r"\[\s*[A-Za-z_]\w*\s*(?:\+|-)\s*(?:[A-Za-z_]\w*|\d+)\s*\]",
        analysed,
    )
    if len(neighbour_accesses) >= 2:
        return True

    lowered = analysed.casefold()
    has_window_vocabulary = any(
        token in lowered for token in ("tap", "coeff", "window", "shift_reg", "line_buffer")
    )
    has_nested_loops = len(re.findall(r"\bfor\s*\(", analysed)) >= 2
    return has_window_vocabulary and has_nested_loops


def _looks_like_dataflow(source: str, categories: list[str]) -> bool:
    lowered = source.casefold()
    if "hls::stream" in lowered:
        return True
    return any(
        any(token in category for token in ("dataflow", "producer", "consumer", "stream"))
        for category in categories
    )


def _unique_three(preferred: list[str]) -> tuple[str, str, str]:
    ordered: list[str] = []
    for family in [*preferred, *DEFAULT_EXPLORATION_STRATEGY_FAMILIES]:
        if family not in LAYER_ONE_STRATEGY_FAMILIES or family in ordered:
            continue
        ordered.append(family)
        if len(ordered) == 3:
            break
    return tuple(ordered)  # type: ignore[return-value]


def select_exploration_strategy_families(
    output_dir: Path,
) -> tuple[tuple[str, str, str], dict[str, Any]]:
    """Choose three layer-one families from immutable baseline evidence.

    Returns both the ordered strategy tuple and an auditable explanation.  The
    rules intentionally prefer false negatives over speculative transforms.
    """

    hierarchy = _load_json(output_dir / "baseline_hierarchical_diagnosis.json")
    cause = _load_json(output_dir / "baseline_source_cause.json")
    target = _load_json(output_dir / "baseline_source_target.json")
    categories = _diagnosis_categories(hierarchy)

    primary_hypothesis = cause.get("primary_hypothesis")
    cause_category = ""
    cause_confidence = None
    if isinstance(primary_hypothesis, dict):
        raw_category = primary_hypothesis.get("category")
        if isinstance(raw_category, str):
            cause_category = raw_category.casefold()
        raw_confidence = primary_hypothesis.get("confidence")
        if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
            cause_confidence = float(raw_confidence)

    source = _source_text(output_dir, target)
    reasons: list[str] = []
    preferred: list[str] = []

    # Task-level producer/consumer evidence is strong enough to justify an
    # explicit DATAFLOW branch.  This rule is deliberately checked first.
    if _looks_like_dataflow(source, categories):
        preferred.extend(["dataflow_pipeline", "memory_parallelism", "bounded_unroll"])
        reasons.append("producer/consumer or stream structure supports task-level pipelining")

    # Repeated neighbouring accesses indicate temporal/spatial reuse suitable
    # for shift registers, line buffers or sliding windows.
    elif _looks_like_sliding_window(source):
        preferred.extend(["sliding_window_reuse", "critical_path_restructuring", "bounded_unroll"])
        reasons.append("source contains repeated neighbouring-index/window reuse")

    # Explicit Vitis RAM-port lower-bound evidence is the strongest memory
    # signal.  Explore both direct banking and coordinated buffered parallelism.
    elif "memory_port_contention" in categories:
        preferred.extend(["memory_parallelism", "buffered_parallelism", "critical_path_restructuring"])
        reasons.append("Vitis reports memory-port contention")

    # GEMM-like *structure* is detected from evidence, not benchmark identity:
    # a dominant parent region, child loops already close to their sequential
    # II=1 lower bound, plus source-level memory/port pressure.  Plain unroll is
    # intentionally displaced because the diagnosis already warns that compute
    # replication without matching bandwidth is unlikely to help.
    elif (
        "dominant_latency_region" in categories
        and "near_sequential_lower_bound" in categories
        and cause_category == "memory_access_or_port_pressure"
        and (cause_confidence is None or cause_confidence >= 0.5)
    ):
        preferred.extend(["buffered_parallelism", "critical_path_restructuring", "memory_parallelism"])
        reasons.append(
            "dominant nested region has II=1 children plus memory/port pressure; coordinated reuse and parallelism is preferred over plain unroll"
        )

    # Recurrence/dependency evidence keeps the historical reduction-first path,
    # with buffered parallelism available as the third independent hypothesis.
    elif any(
        any(token in category for token in ("recurrence", "dependency", "critical_path"))
        for category in categories
    ):
        preferred.extend(["critical_path_restructuring", "bounded_unroll", "buffered_parallelism"])
        reasons.append("dependency/recurrence evidence favours reduction restructuring")

    selected = (
        _unique_three(preferred)
        if preferred
        else DEFAULT_EXPLORATION_STRATEGY_FAMILIES
    )
    if not preferred:
        reasons.append("no high-confidence layer-one trigger; preserve historical exploration schedule")

    audit = {
        "schema_version": 1,
        "selected_strategy_families": list(selected),
        "fallback_default": selected == DEFAULT_EXPLORATION_STRATEGY_FAMILIES and not preferred,
        "reasons": reasons,
        "evidence": {
            "hierarchical_categories": categories,
            "source_cause_category": cause_category or None,
            "source_cause_confidence": cause_confidence,
            "sliding_window_detected": _looks_like_sliding_window(source),
            "dataflow_detected": _looks_like_dataflow(source, categories),
        },
    }
    return selected, audit


def resolve_exploration_strategy_families(
    output_dir: Path,
) -> tuple[str, str, str]:
    """Return one immutable per-run exploration plan and persist its evidence."""

    plan_path = output_dir / PLAN_FILE
    existing = _load_json(plan_path)
    selected = existing.get("selected_strategy_families")
    if (
        isinstance(selected, list)
        and len(selected) == 3
        and len(set(selected)) == 3
        and all(isinstance(item, str) and item in LAYER_ONE_STRATEGY_FAMILIES for item in selected)
    ):
        return tuple(selected)  # type: ignore[return-value]

    families, audit = select_exploration_strategy_families(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return families
