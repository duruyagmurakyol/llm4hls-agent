#!/usr/bin/env python3

"""Run repeated staged one-shot versus iterative repairs for one model."""

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
BENCHMARKS = ("bicg", "atax")
CASES = (
    "staged_compile_then_functional",
    "staged_interface_then_functional",
    "staged_compile_compile_functional",
)
MODES = (("one_shot", 1), ("iterative", 3))


def result_directories(experiment_id: str) -> set[Path]:
    root = ROOT / "runs/experiments" / experiment_id
    if not root.exists():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def newest_result(experiment_id: str, before: set[Path]) -> Path:
    after = result_directories(experiment_id)
    created = sorted(after - before)
    if len(created) != 1:
        raise RuntimeError(
            f"Expected exactly one new result directory for {experiment_id}; found {len(created)}"
        )
    result_path = created[0] / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"Missing result: {result_path}")
    return result_path


def run_once(
    benchmark: str,
    case: str,
    mode: str,
    max_iterations: int,
    repetition: int,
    model_slug: str,
) -> dict[str, Any]:
    config = ROOT / f"configs/{benchmark}_iterative_{model_slug}/{case}.json"
    if not config.is_file():
        raise RuntimeError(f"Missing config: {config}")
    cfg = json.loads(config.read_text(encoding="utf-8"))
    experiment_id = str(cfg["experiment_id"])
    before = result_directories(experiment_id)
    print(
        f"\n[{benchmark}:{case}] {mode} repetition {repetition}",
        flush=True,
    )
    process = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(config),
            "--max-iterations",
            str(max_iterations),
        ],
        cwd=ROOT,
        text=True,
        check=False,
    )
    result_path = newest_result(experiment_id, before)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    iterations = result.get("iterations", [])
    return {
        "benchmark": benchmark.upper(),
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


def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 3)


def stdev(values: list[float]) -> float:
    return round(statistics.stdev(values), 3) if len(values) > 1 else 0.0


def aggregate(rows: list[dict[str, Any]], benchmark: str, mode: str) -> dict[str, Any]:
    selected = [
        row for row in rows
        if (benchmark == "ALL" or row["benchmark"] == benchmark) and row["mode"] == mode
    ]
    passed = [row for row in selected if row["passed"]]
    tokens = [float(row["total_tokens"]) for row in selected]
    latency = [float(row["total_api_latency_seconds"]) for row in selected]
    iterations = [float(row["iterations_executed"]) for row in selected]
    distribution = Counter(str(row["success_iteration"]) for row in passed)
    return {
        "benchmark": benchmark,
        "mode": mode,
        "runs": len(selected),
        "passed_runs": len(passed),
        "success_rate": round(len(passed) / len(selected), 4),
        "mean_tokens": mean(tokens),
        "stdev_tokens": stdev(tokens),
        "mean_latency_seconds": mean(latency),
        "stdev_latency_seconds": stdev(latency),
        "mean_iterations": mean(iterations),
        "success_iteration_distribution": dict(sorted(distribution.items())),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("repetitions must be at least 1")

    manifest_path = ROOT / f"configs/staged_model_{args.model_slug}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / f"results/ablations/staged_{args.model_slug}_repeated" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    try:
        for repetition in range(1, args.repetitions + 1):
            for benchmark in BENCHMARKS:
                for case in CASES:
                    for mode, max_iterations in MODES:
                        rows.append(
                            run_once(
                                benchmark,
                                case,
                                mode,
                                max_iterations,
                                repetition,
                                args.model_slug,
                            )
                        )
                        write_csv(output_dir / "runs_partial.csv", rows)
    except Exception:
        print(f"Partial results preserved in {output_dir.relative_to(ROOT)}")
        raise

    aggregates = [
        aggregate(rows, benchmark, mode)
        for benchmark in ("BICG", "ATAX", "ALL")
        for mode, _ in MODES
    ]
    anomalies = []
    for row in rows:
        reasons = []
        if row["mode"] == "one_shot" and row["passed"]:
            reasons.append("one_shot_passed")
        if row["mode"] == "iterative" and not row["passed"]:
            reasons.append("iterative_failed")
        if row["passed"] and not row["independent_validation_passed"]:
            reasons.append("independent_validation_failed")
        if not row["protected_files_unchanged"]:
            reasons.append("protected_file_changed")
        if reasons:
            anomalies.append({**row, "reasons": reasons})

    summary = {
        "schema_version": 1,
        "ablation": f"staged_{args.model_slug}_repeated",
        "timestamp_utc": timestamp,
        "model": manifest["model"],
        "model_slug": args.model_slug,
        "repetitions": args.repetitions,
        "total_runs": len(rows),
        "aggregates": aggregates,
        "anomalies": anomalies,
        "runs": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "aggregates.csv", aggregates)
    (output_dir / "anomalies.json").write_text(
        json.dumps(anomalies, indent=2) + "\n", encoding="utf-8"
    )

    print("\nRepeated staged-model summary")
    for item in aggregates:
        print(
            f"{item['benchmark']} {item['mode']}: "
            f"{item['passed_runs']}/{item['runs']} passed, "
            f"tokens={item['mean_tokens']}±{item['stdev_tokens']}, "
            f"latency={item['mean_latency_seconds']}±{item['stdev_latency_seconds']}s"
        )
    print(f"Anomalies: {len(anomalies)}")
    print(f"Results: {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
