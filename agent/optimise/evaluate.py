"""Evaluate HLS PPA candidates and maintain a benchmark-independent Pareto archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agent.optimise.duplicate import normalise_source
from agent.optimise.metrics import (
    comparison_metric,
    derive_hardware_metrics,
    maximum_clock_period_ns,
    metric_delta_percent,
)
from agent.optimise.selection import (
    candidate_cost,
    evaluate_resource_limits,
    is_fully_verified,
)
from agent.state import SynthesisMetrics
from agent.tools.synthesis import parse_csynth_xml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINIMUM_FREQUENCY_MHZ = 100.0
PROMISING_RESOURCE_REDUCTION_PERCENT = 25.0
METRIC_KEYS = (
    "clock_period_ns",
    "frequency_mhz",
    "minimum_frequency_mhz",
    "maximum_clock_period_ns",
    "latency_best_cycles",
    "latency_average_cycles",
    "latency_worst_cycles",
    "latency_ns",
    "latency_best_ns",
    "latency_average_ns",
    "latency_worst_ns",
    "interval_min_cycles",
    "interval_max_cycles",
    "throughput_period_ns",
    "throughput_period_min_ns",
    "throughput_period_max_ns",
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
)
OBJECTIVES = (
    "latency_ns",
    "throughput_period_ns",
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
)
PARETO_ELIGIBLE_VERDICTS = {"accept_dominates_baseline", "keep_pareto_candidate"}


def load_json(path: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected in {path}")
    return value


def _actual_or_cycles(metrics: SynthesisMetrics) -> tuple[float | int | None, float | int | None]:
    if metrics.clock_period_ns is not None and metrics.clock_period_ns > 0:
        latency = (
            metrics.latency_cycles * metrics.clock_period_ns
            if metrics.latency_cycles is not None
            else None
        )
        interval = (
            metrics.interval_cycles * metrics.clock_period_ns
            if metrics.interval_cycles is not None
            else None
        )
        return latency, interval
    return metrics.latency_cycles, metrics.interval_cycles


def dominates(left: SynthesisMetrics, right: SynthesisMetrics) -> bool:
    left_latency, left_interval = _actual_or_cycles(left)
    right_latency, right_interval = _actual_or_cycles(right)
    left_values = (left_latency, left_interval, left.lut, left.ff, left.dsp, left.bram)
    right_values = (right_latency, right_interval, right.lut, right.ff, right.dsp, right.bram)
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
    """Use the same comment/whitespace-insensitive identity as the early duplicate gate."""
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(normalise_source(text).encode("utf-8")).hexdigest()


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


def _baseline_project_metrics(config: dict[str, Any]) -> dict[str, Any] | None:
    baseline = config.get("baseline", {})
    project_value = baseline.get("project_dir")
    top = str(config.get("top_function", "")).strip()
    if not project_value or not top:
        return None
    project_dir = Path(str(project_value))
    if not project_dir.is_absolute():
        project_dir = REPO_ROOT / project_dir
    if not project_dir.is_dir():
        return None
    exact = sorted(project_dir.rglob(f"{top}_csynth.xml"))
    candidates = exact or sorted(project_dir.rglob("*_csynth.xml"))
    for path in candidates:
        try:
            metrics = parse_csynth_xml(path)
        except (OSError, ValueError):
            continue
        if any(value is not None for value in metrics.values()):
            return metrics
    return None


def baseline_metrics(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    configured = config.get("baseline", {}).get("metrics")
    if isinstance(configured, dict) and configured:
        return configured

    persisted_path = output_dir / "baseline_metrics.json"
    if persisted_path.is_file():
        persisted = load_json(persisted_path)
        metrics = persisted.get("metrics") if isinstance(persisted.get("metrics"), dict) else persisted
        if isinstance(metrics, dict) and metrics:
            return metrics

    project_metrics = _baseline_project_metrics(config)
    if project_metrics:
        persisted_path.write_text(
            json.dumps({"top_function": config.get("top_function"), "metrics": project_metrics}, indent=2) + "\n",
            encoding="utf-8",
        )
        return project_metrics

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
    return metric_delta_percent(value, baseline)


def load_optional(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.is_file() else None


def _baseline_fully_verified(config: dict[str, Any]) -> bool:
    verification = config.get("baseline", {}).get("verification")
    if not isinstance(verification, dict):
        return True
    return all(
        verification.get(key) is True
        for key in ("csim_passed", "synthesis_passed", "cosim_passed")
    )


def _objective_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def classify_candidate(
    output_dir: Path,
    index: int,
    baseline: dict[str, Any],
    duplicates: dict[int, int],
    *,
    minimum_frequency_mhz: float = DEFAULT_MINIMUM_FREQUENCY_MHZ,
    resource_limits: Any = None,
) -> dict[str, Any]:
    prefix = f"candidate_{index:03d}"
    source = output_dir / f"{prefix}.cpp"
    static = load_optional(output_dir / f"{prefix}_static_validation.json")
    csim = load_optional(output_dir / f"{prefix}_csim_validation.json")
    synthesis = load_optional(output_dir / f"{prefix}_synthesis.json")
    cosim = load_optional(output_dir / f"{prefix}_cosim.json")
    synthesis_attempted = bool(synthesis and synthesis.get("synthesis_run") is True)
    synthesis_completed = bool(
        synthesis
        and synthesis.get("passed") is True
        and isinstance(synthesis.get("metrics"), dict)
        and synthesis.get("metrics")
    )
    cosim_attempted = bool(cosim and cosim.get("cosim_run") is True)
    record: dict[str, Any] = {
        "candidate_index": index,
        "candidate_file": str(source.relative_to(REPO_ROOT)),
        "static_validation": static.get("passed") if static else None,
        "csim": csim.get("passed") if csim else None,
        "synthesis": synthesis.get("passed") if synthesis else None,
        "synthesis_run": synthesis_attempted,
        "cosim": cosim.get("passed") if cosim else None,
        "cosim_run": cosim_attempted,
        "fully_verified": False,
        "metrics": {},
        "deltas_percent": {},
        "performance_comparison": {},
        "usefulness_classification": None,
        "refinement_eligible": False,
        "minimum_frequency_mhz": minimum_frequency_mhz,
        "meets_frequency_requirement": None,
        "resource_limit_compliance": evaluate_resource_limits({}, resource_limits),
        "meets_resource_limits": None,
        "cost": candidate_cost(output_dir, index),
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
        record.update(verdict="reject_csim", reason="Vitis CSim failed or the candidate was not compiled.")
        return record
    if synthesis_attempted and synthesis and synthesis.get("timed_out") is True:
        timeout = synthesis.get("timeout_seconds")
        record.update(
            verdict="reject_synthesis_timeout",
            reason=f"Vitis synthesis exceeded the {timeout}-second timeout and its process group was terminated.",
        )
        return record
    if synthesis_attempted and synthesis and synthesis.get("passed") is False:
        record.update(
            verdict="reject_synthesis_failed",
            reason="Vitis synthesis failed before a valid top-level report was produced.",
        )
        return record
    if not synthesis_completed:
        return record

    metrics = derive_hardware_metrics(
        synthesis.get("metrics") or {},
        minimum_frequency_mhz=minimum_frequency_mhz,
    )
    record["metrics"] = {key: metrics.get(key) for key in METRIC_KEYS}
    record["meets_frequency_requirement"] = metrics.get("meets_minimum_frequency")
    compliance = evaluate_resource_limits(metrics, resource_limits)
    record["resource_limit_compliance"] = compliance
    record["meets_resource_limits"] = compliance["passed"]

    # Preserve baseline-relative evidence as soon as valid synthesis metrics exist.
    # Hard constraints and final verification still control verdicts and eligibility.
    record["deltas_percent"] = {
        key: percent_change(metrics.get(key), baseline.get(key)) for key in METRIC_KEYS
    }
    latency_key, candidate_latency, baseline_latency = comparison_metric(
        metrics,
        baseline,
        actual_key="latency_ns",
        cycle_key="latency_best_cycles",
    )
    interval_key, candidate_interval, baseline_interval = comparison_metric(
        metrics,
        baseline,
        actual_key="throughput_period_ns",
        cycle_key="interval_min_cycles",
    )
    latency_delta = percent_change(candidate_latency, baseline_latency)
    interval_delta = percent_change(candidate_interval, baseline_interval)
    record["performance_comparison"] = {
        "latency_metric": latency_key,
        "latency_candidate": candidate_latency,
        "latency_baseline": baseline_latency,
        "latency_delta_percent": latency_delta,
        "throughput_metric": interval_key,
        "throughput_candidate": candidate_interval,
        "throughput_baseline": baseline_interval,
        "throughput_delta_percent": interval_delta,
    }
    resource_deltas = [
        record["deltas_percent"].get(key)
        for key in (
            "resources_lut_used",
            "resources_ff_used",
            "resources_dsp_used",
            "resources_bram_used",
        )
    ]
    numeric_resources = [
        value for value in resource_deltas if isinstance(value, (int, float))
    ]
    significant_resource_reduction = any(
        value <= -PROMISING_RESOURCE_REDUCTION_PERCENT
        for value in numeric_resources
    )
    constraint_usefulness = (
        "promising_constraint_violation"
        if significant_resource_reduction
        else "constraint_violation"
    )

    frequency = metrics.get("frequency_mhz")
    maximum_period = metrics.get("maximum_clock_period_ns")
    if frequency is None:
        record.update(
            verdict="reject_frequency_unavailable",
            reason=(
                "No valid estimated clock period was available, so compliance with the "
                f"{minimum_frequency_mhz:g} MHz minimum cannot be established."
            ),
        )
        return record
    if metrics.get("meets_minimum_frequency") is not True:
        record.update(
            verdict="reject_frequency_threshold",
            usefulness_classification=constraint_usefulness,
            refinement_eligible=significant_resource_reduction,
            reason=(
                f"Estimated frequency {frequency:.3f} MHz is below the required "
                f"{minimum_frequency_mhz:g} MHz (clock period must be at most "
                f"{maximum_period:.3f} ns)."
            ),
        )
        return record
    if compliance["passed"] is not True:
        labels = ", ".join(
            f"{item['metric']}={item.get('actual')} > {item['limit']}"
            for item in compliance["violations"]
        )
        record.update(
            verdict="reject_resource_limits",
            usefulness_classification=constraint_usefulness,
            refinement_eligible=significant_resource_reduction,
            reason="Candidate violates task-specific resource limits: " + labels,
        )
        return record

    if cosim_attempted and cosim and cosim.get("timed_out") is True:
        record.update(
            verdict="reject_cosim_timeout",
            reason=f"C/RTL co-simulation exceeded the {cosim.get('timeout_seconds')}-second timeout.",
        )
        return record
    if cosim_attempted and cosim and cosim.get("passed") is False:
        record.update(
            verdict="reject_cosim",
            reason="C/RTL co-simulation failed; the candidate is not fully verified.",
        )
        return record
    if not cosim_attempted:
        record.update(
            verdict="awaiting_cosim",
            reason="Synthesis, frequency and resource checks passed; C/RTL co-simulation is required.",
        )
        return record

    record["fully_verified"] = bool(
        record["static_validation"] is True
        and record["csim"] is True
        and record["synthesis"] is True
        and record["cosim"] is True
    )
    if not is_fully_verified(record):
        record.update(
            verdict="reject_not_fully_verified",
            reason="Candidate is missing one or more required verification stages.",
        )
        return record

    objective_pairs = [
        (latency_key, candidate_latency, baseline_latency),
        (interval_key, candidate_interval, baseline_interval),
        *[
            (key, metrics.get(key), baseline.get(key))
            for key in OBJECTIVES[2:]
        ],
    ]
    missing_objectives = [
        key
        for key, candidate_value, baseline_value in objective_pairs
        if not (
            _objective_number(candidate_value)
            and _objective_number(baseline_value)
        )
    ]
    if missing_objectives:
        record.update(
            verdict="reject_objective_metrics_unavailable",
            missing_objectives=missing_objectives,
            reason=(
                "Required objective metrics are unavailable for comparison: "
                + ", ".join(missing_objectives)
                + "."
            ),
        )
        return record

    candidate_objectives = [float(candidate_value) for _, candidate_value, _ in objective_pairs]
    baseline_objectives = [float(baseline_value) for _, _, baseline_value in objective_pairs]
    same_performance = candidate_objectives[:2] == baseline_objectives[:2]
    same_resources = candidate_objectives[2:] == baseline_objectives[2:]
    no_objective_increase = all(
        candidate_value <= baseline_value
        for candidate_value, baseline_value in zip(
            candidate_objectives,
            baseline_objectives,
        )
    )
    any_improvement = any(
        candidate_value < baseline_value
        for candidate_value, baseline_value in zip(
            candidate_objectives,
            baseline_objectives,
        )
    )

    if same_performance and same_resources:
        record.update(verdict="reject_no_change", reason="Synthesis metrics are identical to the baseline.")
    elif any_improvement and no_objective_increase:
        record.update(
            verdict="accept_dominates_baseline",
            reason="Fully verified candidate improves at least one objective without worsening another.",
        )
    elif any_improvement:
        record.update(
            verdict="keep_pareto_candidate",
            reason="Fully verified candidate offers a performance or resource trade-off.",
        )
    else:
        record.update(
            verdict="reject_no_objective_gain",
            reason="Fully verified candidate does not improve latency, throughput or measured resources.",
        )
    return record


def objective_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else float("inf")


def record_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        objective_value(left["metrics"], key) <= objective_value(right["metrics"], key)
        for key in OBJECTIVES
    ) and any(
        objective_value(left["metrics"], key) < objective_value(right["metrics"], key)
        for key in OBJECTIVES
    )


def evaluate_experiment(config_path: Any) -> dict[str, Any]:
    config = load_json(config_path.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = candidate_indices(output_dir)
    minimum_frequency = float(
        config.get("minimum_frequency_mhz", DEFAULT_MINIMUM_FREQUENCY_MHZ)
    )
    resource_limits = config.get("resource_limits") or {}
    baseline = derive_hardware_metrics(
        baseline_metrics(config, output_dir),
        minimum_frequency_mhz=minimum_frequency,
    )
    baseline_compliance = evaluate_resource_limits(baseline, resource_limits)
    duplicates = duplicate_map(output_dir, indices)
    records = [
        classify_candidate(
            output_dir,
            index,
            baseline,
            duplicates,
            minimum_frequency_mhz=minimum_frequency,
            resource_limits=resource_limits,
        )
        for index in indices
    ]
    pareto = [
        record
        for record in records
        if record["verdict"] in PARETO_ELIGIBLE_VERDICTS
    ]
    for record in pareto:
        dominator = next(
            (
                other
                for other in pareto
                if other["candidate_index"] != record["candidate_index"]
                and record_dominates(other, record)
            ),
            None,
        )
        if dominator:
            record.update(
                verdict="reject_dominated",
                reason=f"Candidate is dominated by candidate_{dominator['candidate_index']:03d}.",
                dominated_by=dominator["candidate_index"],
            )

    final_pareto = [record for record in pareto if record["verdict"] in PARETO_ELIGIBLE_VERDICTS]
    summary = {
        "schema_version": 7,
        "experiment_name": config.get("experiment_name"),
        "benchmark": config.get("benchmark"),
        "selection": config.get("selection", {}),
        "frequency_requirement": {
            "minimum_frequency_mhz": minimum_frequency,
            "maximum_clock_period_ns": maximum_clock_period_ns(minimum_frequency),
            "baseline_frequency_mhz": baseline.get("frequency_mhz"),
            "baseline_meets_requirement": baseline.get("meets_minimum_frequency"),
        },
        "resource_limits": {
            "configured": baseline_compliance["configured"],
            "limits": baseline_compliance["limits"],
            "baseline_compliance": baseline_compliance,
        },
        "baseline_metrics": {key: baseline.get(key) for key in METRIC_KEYS},
        "baseline_record": {
            "candidate_index": 0,
            "candidate_file": config["baseline"]["source"],
            "metrics": {key: baseline.get(key) for key in METRIC_KEYS},
            "static_validation": True,
            "csim": True,
            "synthesis": True,
            "cosim": True,
            "fully_verified": _baseline_fully_verified(config),
            "meets_frequency_requirement": baseline.get("meets_minimum_frequency"),
            "resource_limit_compliance": baseline_compliance,
            "meets_resource_limits": baseline_compliance["passed"],
            "verdict": "baseline",
            "cost": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "tool_calls": 0,
                "tool_seconds": 0.0,
            },
        },
        "candidates": records,
        "pareto_archive": [
            {
                "candidate_index": 0,
                "candidate_file": config["baseline"]["source"],
                "metrics": {key: baseline.get(key) for key in METRIC_KEYS},
                "fully_verified": _baseline_fully_verified(config),
                "meets_frequency_requirement": baseline.get("meets_minimum_frequency"),
                "resource_limit_compliance": baseline_compliance,
                "meets_resource_limits": baseline_compliance["passed"],
                "verdict": "baseline",
                "cost": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "tool_calls": 0,
                    "tool_seconds": 0.0,
                },
            },
            *final_pareto,
        ],
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HLS PPA candidates.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    summary = evaluate_experiment(args.config)
    print("\nPPA candidate evaluation")
    print(f"Experiment: {summary.get('experiment_name')}")
    print(f"Candidates: {len(summary['candidates'])}")
    print(f"Pareto archive size: {len(summary['pareto_archive'])}")
    for candidate in summary["candidates"]:
        print(
            f"Candidate {candidate['candidate_index']:03d}: {candidate['verdict']} — {candidate['reason']}"
        )


if __name__ == "__main__":
    main()
