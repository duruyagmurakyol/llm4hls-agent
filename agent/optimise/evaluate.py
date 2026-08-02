"""Evaluate HLS PPA candidates and maintain a benchmark-independent Pareto archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agent.state import SynthesisMetrics

REPO_ROOT = Path(__file__).resolve().parents[2]
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
PARETO_ELIGIBLE_VERDICTS = {"accept_dominates_baseline", "keep_pareto_candidate"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dominates(left: SynthesisMetrics, right: SynthesisMetrics) -> bool:
    left_values = (left.latency_cycles, left.lut, left.ff, left.dsp, left.bram)
    right_values = (right.latency_cycles, right.lut, right.ff, right.dsp, right.bram)
    if any(value is None for value in (*left_values, *right_values)):
        return False
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def candidate_indices(output_dir: Path) -> list[int]:
    pattern = re.compile(r"candidate_(\d{3})\.cpp$")
    return sorted(
        {
            int(match.group(1))
            for path in output_dir.glob("candidate_*.cpp")
            if (match := pattern.match(path.name))
        }
    )


def normalised_source_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalised = "\n".join(line.rstrip() for line in text.splitlines()).strip() + "\n"
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def duplicate_map(output_dir: Path, indices: list[int]) -> dict[int, int]:
    first_by_hash: dict[str, int] = {}
    duplicates: dict[int, int] = {}
    for index in indices:
        digest = normalised_source_hash(output_dir / f"candidate_{index:03d}.cpp")
        if digest in first_by_hash:
            duplicates[index] = first_by_hash[digest]
        else:
            first_by_hash[digest] = index
    return duplicates


def baseline_metrics(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    configured = config.get("baseline", {}).get("metrics")
    if isinstance(configured, dict) and configured:
        return configured
    diagnosis_path = output_dir / "baseline_hierarchical_diagnosis.json"
    if diagnosis_path.is_file():
        diagnosis = load_json(diagnosis_path)
        for item in (
            diagnosis.get("top_function"),
            diagnosis.get("top_level"),
            diagnosis.get("recommended_focus"),
        ):
            if isinstance(item, dict) and isinstance(item.get("metrics"), dict) and item["metrics"]:
                return item["metrics"]
        for key in ("reports", "functions", "ranked_targets"):
            for item in diagnosis.get(key, []) if isinstance(diagnosis.get(key), list) else []:
                if isinstance(item, dict) and isinstance(item.get("metrics"), dict) and item["metrics"]:
                    return item["metrics"]
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


def classify_candidate(
    output_dir: Path,
    index: int,
    baseline: dict[str, Any],
    duplicates: dict[int, int],
) -> dict[str, Any]:
    prefix = f"candidate_{index:03d}"
    source = output_dir / f"{prefix}.cpp"
    static = load_optional(output_dir / f"{prefix}_static_validation.json")
    csim = load_optional(output_dir / f"{prefix}_csim_validation.json")
    synthesis = load_optional(output_dir / f"{prefix}_synthesis.json")
    synthesis_attempted = bool(synthesis and synthesis.get("synthesis_run") is True)
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
        "synthesis_run": synthesis_attempted,
        "metrics": {},
        "deltas_percent": {},
        "verdict": "incomplete",
        "reason": "Candidate has not completed all required evaluation stages.",
    }
    if index in duplicates:
        record.update(
            duplicate_of=duplicates[index],
            verdict="reject_duplicate",
            reason=f"Source is identical to candidate_{duplicates[index]:03d}.",
        )
        return record
    if static and static.get("passed") is False:
        failed = [name for name, passed in (static.get("checks") or {}).items() if not passed]
        record.update(verdict="reject_static", reason="Static validation failed: " + ", ".join(failed))
        return record
    if csim and csim.get("passed") is False:
        verdict = "reject_csim_timeout" if csim.get("timed_out") is True else "reject_csim"
        reason = (
            "Vitis CSim exceeded its timeout."
            if verdict == "reject_csim_timeout"
            else "Vitis CSim failed or the candidate was not compiled."
        )
        record.update(verdict=verdict, reason=reason)
        return record
    if synthesis and synthesis.get("timed_out") is True:
        record.update(
            verdict="reject_synthesis_timeout",
            reason=f"Vitis synthesis exceeded {synthesis.get('timeout_seconds')} seconds.",
        )
        return record
    if synthesis_attempted and synthesis and synthesis.get("passed") is False:
        record.update(
            verdict="reject_synthesis_failed",
            reason="Vitis synthesis failed before producing complete top-level metrics.",
        )
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
    resource_deltas = [
        record["deltas_percent"].get(key)
        for key in ("resources_lut_used", "resources_ff_used", "resources_dsp_used", "resources_bram_used")
    ]
    numeric_resources = [value for value in resource_deltas if isinstance(value, (int, float))]
    improves_latency = isinstance(latency_delta, (int, float)) and latency_delta < 0
    improves_interval = isinstance(interval_delta, (int, float)) and interval_delta < 0
    same_performance = latency_delta == 0 and interval_delta == 0
    no_resource_increase = all(value <= 0 for value in numeric_resources)
    if same_performance and all(value == 0 for value in numeric_resources):
        record.update(verdict="reject_no_change", reason="Synthesis metrics are identical to the baseline.")
    elif (improves_latency or improves_interval) and no_resource_increase:
        record.update(verdict="accept_dominates_baseline", reason="Improves performance without increasing measured resources.")
    elif improves_latency or improves_interval:
        record.update(verdict="keep_pareto_candidate", reason="Improves performance with a resource trade-off.")
    else:
        record.update(verdict="reject_no_performance_gain", reason="Does not improve latency or interval over the baseline.")
    return record


def objective_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else float("inf")


def record_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        objective_value(left["metrics"], key) <= objective_value(right["metrics"], key)
        for key in OBJECTIVES
    ) and any(
        objective_value(left["metrics"], key) < objective_value(right["metrics"], key)
        for key in OBJECTIVES
    )


def evaluate_experiment(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = candidate_indices(output_dir)
    baseline = baseline_metrics(config, output_dir)
    duplicates = duplicate_map(output_dir, indices)
    records = [classify_candidate(output_dir, index, baseline, duplicates) for index in indices]
    eligible = [record for record in records if record.get("verdict") in PARETO_ELIGIBLE_VERDICTS]
    baseline_record = {
        "candidate_index": 0,
        "candidate_file": config["baseline"]["source"],
        "metrics": {key: baseline.get(key) for key in METRIC_KEYS},
        "verdict": "baseline",
    }
    pool = [baseline_record, *eligible]
    pareto = [
        item
        for item in pool
        if not any(other is not item and record_dominates(other, item) for other in pool)
    ]
    synthesis_calls_used = sum(1 for record in records if record.get("synthesis_run"))
    maximum = int(config["budget"]["max_synthesis_calls"])
    summary = {
        "experiment_name": config["experiment_name"],
        "benchmark": config["benchmark"],
        "baseline_metrics": {key: baseline.get(key) for key in METRIC_KEYS},
        "budget": {
            "max_candidates": config["budget"]["max_candidates"],
            "max_synthesis_calls": maximum,
            "synthesis_calls_used": synthesis_calls_used,
            "synthesis_calls_remaining": max(0, maximum - synthesis_calls_used),
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
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPA candidates and build a Pareto archive.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    summary = evaluate_experiment(args.config)
    print("\nPPA experiment evaluation")
    print(f"Benchmark: {summary['benchmark']}")
    print(f"Candidates found: {len(summary['candidates'])}")
    budget = summary["budget"]
    print(f"Synthesis calls used: {budget['synthesis_calls_used']}/{budget['max_synthesis_calls']}")
    print("\nCandidate verdicts")
    for record in summary["candidates"]:
        print(f"  {record['candidate_index']:03d}: {record['verdict']} — {record['reason']}")
    print("\nPareto archive")
    for item in summary["pareto_archive"]:
        label = "baseline" if item["candidate_index"] == 0 else f"candidate_{item['candidate_index']:03d}"
        metrics = item["metrics"]
        print(
            f"  {label}: latency={metrics.get('latency_best_cycles')}, "
            f"LUT={metrics.get('resources_lut_used')}, "
            f"FF={metrics.get('resources_ff_used')}, "
            f"DSP={metrics.get('resources_dsp_used')}"
        )
    print("No model call, CSim, or synthesis was run.")


if __name__ == "__main__":
    main()
