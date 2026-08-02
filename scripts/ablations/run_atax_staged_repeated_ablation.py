#!/usr/bin/env python3

"""Run repeated one-shot versus iterative staged ATAX repair experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/experiments/run_iterative_api_experiment.py"
CONFIG_ROOT = ROOT / "configs/atax_iterative_qwen35"
CASES = [
    "staged_compile_then_functional",
    "staged_interface_then_functional",
    "staged_compile_compile_functional",
]
MODES = [("one_shot", 1), ("iterative", 3)]


def experiment_id(case: str) -> str:
    return f"atax_iterative_qwen35_{case}"


def result_directories(case: str) -> set[Path]:
    root = ROOT / "runs/experiments" / experiment_id(case)
    if not root.exists():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def new_result_path(case: str, before: set[Path]) -> Path:
    created = sorted(result_directories(case) - before)
    if len(created) != 1:
        raise RuntimeError(f"Expected one new result for {case}; found {len(created)}")
    result = created[0] / "result.json"
    if not result.is_file():
        raise RuntimeError(f"Missing result file: {result}")
    return result


def run_once(case: str, mode: str, max_iterations: int, repetition: int) -> dict[str, Any]:
    config = CONFIG_ROOT / f"{case}.json"
    before = result_directories(case)
    command = [
        sys.executable,
        str(RUNNER),
        str(config),
        "--max-iterations",
        str(max_iterations),
    ]
    print(f"\n[{case}] {mode} repetition {repetition}", flush=True)
    process = subprocess.run(command, cwd=ROOT, text=True, check=False)
    result_path = new_result_path(case, before)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    iterations = result.get("iterations", [])
    return {
        "case": case,
        "mode": mode,
        "repetition": repetition,
        "max_iterations": max_iterations,
        "runner_return_code": process.returncode,
        "passed": bool(result.get("passed", False)),
        "iterations_executed": int(result.get("iterations_executed", 0)),
        "success_iteration": result.get("success_iteration"),
        "total_tokens": result.get("total_tokens"),
        "total_api_latency_seconds": result.get("total_api_latency_seconds"),
        "post_host_validation_passed": bool(result.get("post_host_validation_passed", False)),
        "independent_validation_passed": bool(result.get("independent_validation_passed", False)),
        "protected_files_unchanged": bool(result.get("protected_files_unchanged", False)),
        "failure_sequence": ">".join(
            str(item.get("failure_class_before", "unknown")) for item in iterations
        ),
        "result_path": str(result_path.relative_to(ROOT)),
    }


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def mean_or_none(values: list[Any]) -> float | None:
    xs = numeric(values)
    return round(statistics.mean(xs), 3) if xs else None


def stdev_or_none(values: list[Any]) -> float | None:
    xs = numeric(values)
    return round(statistics.stdev(xs), 3) if len(xs) >= 2 else 0.0 if xs else None


def aggregate(selected: list[dict[str, Any]], case: str, mode: str) -> dict[str, Any]:
    passed = [row for row in selected if row["passed"]]
    distribution = Counter(str(row["success_iteration"]) for row in passed)
    return {
        "case": case,
        "mode": mode,
        "runs": len(selected),
        "passed_runs": len(passed),
        "success_rate": round(len(passed) / len(selected), 4) if selected else None,
        "mean_tokens": mean_or_none([row["total_tokens"] for row in selected]),
        "stdev_tokens": stdev_or_none([row["total_tokens"] for row in selected]),
        "mean_latency_seconds": mean_or_none([row["total_api_latency_seconds"] for row in selected]),
        "stdev_latency_seconds": stdev_or_none([row["total_api_latency_seconds"] for row in selected]),
        "mean_iterations": mean_or_none([row["iterations_executed"] for row in selected]),
        "success_iteration_distribution": dict(sorted(distribution.items())),
    }


def detect_anomalies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if row["mode"] == "one_shot" and row["passed"]:
            reasons.append("one_shot_unexpectedly_passed")
        if row["mode"] == "iterative" and not row["passed"]:
            reasons.append("iterative_failed")
        if row["passed"] and not row["independent_validation_passed"]:
            reasons.append("host_passed_but_independent_failed")
        if not row["protected_files_unchanged"]:
            reasons.append("protected_file_changed")
        if (row["runner_return_code"] == 0) != row["passed"]:
            reasons.append("runner_status_inconsistent")
        if reasons:
            anomalies.append({
                "case": row["case"],
                "mode": row["mode"],
                "repetition": row["repetition"],
                "reasons": reasons,
                "result_path": row["result_path"],
            })
    return anomalies


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 2:
        raise SystemExit("Use at least two repetitions.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "results/ablations/atax_staged_feedback_repeated" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []

    try:
        for repetition in range(1, args.repetitions + 1):
            for case in CASES:
                for mode, max_iterations in MODES:
                    rows.append(run_once(case, mode, max_iterations, repetition))
                    write_csv(output_dir / "runs_partial.csv", rows)
                    (output_dir / "runs_partial.json").write_text(
                        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
                    )
    except Exception:
        print(f"Partial results preserved in: {output_dir.relative_to(ROOT)}")
        raise

    aggregates: list[dict[str, Any]] = []
    for case in CASES:
        for mode, _ in MODES:
            selected = [row for row in rows if row["case"] == case and row["mode"] == mode]
            aggregates.append(aggregate(selected, case, mode))

    overall: list[dict[str, Any]] = []
    for mode, _ in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        overall.append(aggregate(selected, "ALL_CASES", mode))

    anomalies = detect_anomalies(rows)
    summary = {
        "schema_version": 1,
        "ablation": "atax_staged_feedback_repeated",
        "timestamp_utc": timestamp,
        "model": "Qwen/Qwen3.5-122B-A10B",
        "repetitions": args.repetitions,
        "cases": CASES,
        "total_runs": len(rows),
        "aggregates_by_case": aggregates,
        "overall": overall,
        "anomalies": anomalies,
        "runs": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "aggregates.csv", aggregates + overall)
    (output_dir / "anomalies.json").write_text(json.dumps(anomalies, indent=2) + "\n", encoding="utf-8")

    print("\nOverall repeated ATAX summary")
    print("mode | passed/runs | success | mean tokens ± sd | mean latency ± sd")
    print("--- | --- | --- | --- | ---")
    for item in overall:
        print(
            f"{item['mode']} | {item['passed_runs']}/{item['runs']} | "
            f"{100 * item['success_rate']:.1f}% | "
            f"{item['mean_tokens']} ± {item['stdev_tokens']} | "
            f"{item['mean_latency_seconds']} ± {item['stdev_latency_seconds']}"
        )
    print(f"Anomalies: {len(anomalies)}")
    print(f"Results: {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
