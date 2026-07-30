#!/usr/bin/env python3

"""Evaluate an HLS PPA experiment and build a candidate/Pareto summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = (
    "clock_period_ns",
    "latency_best_cycles",
    "latency_average_cycles",
    "latency_worst_cycles",
    "interval_min_cycles",
    "interval_max_cycles",
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
)
OBJECTIVES = (
    "latency_best_cycles",
    "interval_min_cycles",
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
)
PARETO_ELIGIBLE_VERDICTS = {
    "accept_dominates_baseline",
    "keep_pareto_candidate",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_indices(output_dir: Path) -> list[int]:
    indices: set[int] = set()
    pattern = re.compile(r"candidate_(\d{3})\.cpp$")
    for path in output_dir.glob("candidate_*.cpp"):
        match = pattern.match(path.name)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def baseline_metrics(output_dir: Path) -> dict[str, Any]:
    diagnosis_path = output_dir / "baseline_hierarchical_diagnosis.json"
    diagnosis = load_json(diagnosis_path)

    candidates = [
        diagnosis.get("top_function"),
        diagnosis.get("top_level"),
        diagnosis.get("recommended_focus"),
    ]
    for item in candidates:
        if isinstance(item, dict):
            metrics = item.get("metrics")
            if isinstance(metrics, dict) and metrics:
                return metrics

    for key in ("reports", "functions", "ranked_targets"):
        items = diagnosis.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("function") or item.get("name") or item.get("report_name")
            metrics = item.get("metrics")
            if name == "kernel_atax" and isinstance(metrics, dict) and metrics:
                return metrics

    candidate_one = output_dir / "candidate_001_synthesis.json"
    if candidate_one.is_file():
        metrics = load_json(candidate_one).get("metrics")
        if isinstance(metrics, dict) and metrics:
            return metrics

    raise ValueError("Could not locate baseline top-level metrics.")


def percent_change(value: Any, baseline: Any) -> float | None:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    if baseline == 0:
        return 0.0 if value == 0 else None
    return ((value - baseline) / baseline) * 100.0


def load_optional(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.is_file() else None


def classify_candidate(output_dir: Path, index: int, baseline: dict[str, Any]) -> dict[str, Any]:
    prefix = f"candidate_{index:03d}"
    source = output_dir / f"{prefix}.cpp"
    static = load_optional(output_dir / f"{prefix}_static_validation.json")
    csim = load_optional(output_dir / f"{prefix}_csim_validation.json")
    synthesis = load_optional(output_dir / f"{prefix}_synthesis.json")

    synthesis_completed = bool(
        synthesis
        and synthesis.get("passed") is True
        and isinstance(synthesis.get("metrics"), dict)
        and synthesis.get("metrics")
    )

    record: dict[str, Any] = {
        "candidate_index": index,
        "candidate_file": str(source.relative_to(REPO_ROOT)),
        "static_validation": static.get("passed") if static else None,
        "csim": csim.get("passed") if csim else None,
        "synthesis": synthesis.get("passed") if synthesis else None,
        "synthesis_run": synthesis_completed,
        "metrics": {},
        "deltas_percent": {},
        "verdict": "incomplete",
        "reason": "Candidate has not completed all required evaluation stages.",
    }

    if static and static.get("passed") is False:
        failed = [name for name, passed in (static.get("checks") or {}).items() if not passed]
        record["verdict"] = "reject_static"
        record["reason"] = "Static validation failed: " + ", ".join(failed)
        return record

    if csim and csim.get("passed") is False:
        record["verdict"] = "reject_csim"
        record["reason"] = "Vitis CSim failed or the candidate was not compiled."
        return record

    if not synthesis_completed:
        return record

    metrics = synthesis.get("metrics") or {}
    record["metrics"] = {key: metrics.get(key) for key in METRIC_KEYS}
    record["deltas_percent"] = {
        key: percent_change(metrics.get(key), baseline.get(key)) for key in METRIC_KEYS
    }

    latency_delta = record["deltas_percent"].get("latency_best_cycles")
    interval_delta = record["deltas_percent"].get("interval_min_cycles")
    resource_keys = (
        "resources_lut_used",
        "resources_ff_used",
        "resources_dsp_used",
        "resources_bram_used",
    )
    resource_deltas = [record["deltas_percent"].get(key) for key in resource_keys]
    numeric_resource_deltas = [x for x in resource_deltas if isinstance(x, (int, float))]

    improves_latency = isinstance(latency_delta, (int, float)) and latency_delta < 0
    improves_interval = isinstance(interval_delta, (int, float)) and interval_delta < 0
    same_performance = latency_delta == 0 and interval_delta == 0
    no_resource_increase = all(delta <= 0 for delta in numeric_resource_deltas)

    if same_performance and all(delta == 0 for delta in numeric_resource_deltas):
        record["verdict"] = "reject_no_change"
        record["reason"] = "Synthesis metrics are identical to the baseline."
    elif (improves_latency or improves_interval) and no_resource_increase:
        record["verdict"] = "accept_dominates_baseline"
        record["reason"] = "Improves performance without increasing measured resources."
    elif improves_latency or improves_interval:
        record["verdict"] = "keep_pareto_candidate"
        record["reason"] = "Improves performance with a resource trade-off."
    else:
        record["verdict"] = "reject_no_performance_gain"
        record["reason"] = "Does not improve latency or interval over the baseline."

    return record


def objective_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return float("inf")


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_metrics = a["metrics"]
    b_metrics = b["metrics"]
    no_worse = all(
        objective_value(a_metrics, key) <= objective_value(b_metrics, key)
        for key in OBJECTIVES
    )
    strictly_better = any(
        objective_value(a_metrics, key) < objective_value(b_metrics, key)
        for key in OBJECTIVES
    )
    return no_worse and strictly_better


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPA candidates and build a Pareto archive.")
    parser.add_argument("config", type=Path, help="PPA optimisation JSON config")
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = baseline_metrics(output_dir)
    records = [classify_candidate(output_dir, index, baseline) for index in candidate_indices(output_dir)]

    pareto_candidates = [
        record for record in records if record.get("verdict") in PARETO_ELIGIBLE_VERDICTS
    ]
    baseline_record = {
        "candidate_index": 0,
        "candidate_file": config["baseline"]["source"],
        "metrics": {key: baseline.get(key) for key in METRIC_KEYS},
        "verdict": "baseline",
    }
    pareto_pool = [baseline_record, *pareto_candidates]
    pareto = [
        item
        for item in pareto_pool
        if not any(other is not item and dominates(other, item) for other in pareto_pool)
    ]

    synthesis_calls_used = sum(1 for record in records if record.get("synthesis_run"))
    max_synthesis_calls = int(config["budget"]["max_synthesis_calls"])

    summary = {
        "experiment_name": config["experiment_name"],
        "benchmark": config["benchmark"],
        "baseline_metrics": {key: baseline.get(key) for key in METRIC_KEYS},
        "budget": {
            "max_candidates": config["budget"]["max_candidates"],
            "max_synthesis_calls": max_synthesis_calls,
            "synthesis_calls_used": synthesis_calls_used,
            "synthesis_calls_remaining": max(0, max_synthesis_calls - synthesis_calls_used),
        },
        "candidates": records,
        "pareto_archive": [
            {
                "candidate_index": item["candidate_index"],
                "candidate_file": item["candidate_file"],
                "metrics": item["metrics"],
                "verdict": item["verdict"],
            }
            for item in pareto
        ],
    }

    output_path = output_dir / "experiment_summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nPPA experiment evaluation")
    print(f"Benchmark: {config['benchmark']}")
    print(f"Candidates found: {len(records)}")
    print(f"Synthesis calls used: {synthesis_calls_used}/{max_synthesis_calls}")
    print("\nCandidate verdicts")
    for record in records:
        print(f"  {record['candidate_index']:03d}: {record['verdict']} — {record['reason']}")
    print("\nPareto archive")
    for item in pareto:
        label = "baseline" if item["candidate_index"] == 0 else f"candidate_{item['candidate_index']:03d}"
        metrics = item["metrics"]
        print(
            f"  {label}: latency={metrics.get('latency_best_cycles')}, "
            f"LUT={metrics.get('resources_lut_used')}, "
            f"FF={metrics.get('resources_ff_used')}, "
            f"DSP={metrics.get('resources_dsp_used')}"
        )
    print(f"\nSummary: {output_path.relative_to(REPO_ROOT)}")
    print("No model call, CSim, or synthesis was run.")


if __name__ == "__main__":
    main()
