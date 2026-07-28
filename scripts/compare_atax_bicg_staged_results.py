#!/usr/bin/env python3

"""Compare the latest repeated staged-feedback results for ATAX and BICG."""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = {
    "BICG": ROOT / "results/ablations/bicg_staged_feedback_repeated",
    "ATAX": ROOT / "results/ablations/atax_staged_feedback_repeated",
}


def latest_summary(root: Path) -> tuple[Path, dict[str, Any]]:
    if not root.is_dir():
        raise SystemExit(f"Missing results directory: {root.relative_to(ROOT)}")
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise SystemExit(f"No repeated results found in {root.relative_to(ROOT)}")
    summary_path = candidates[-1] / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing summary: {summary_path.relative_to(ROOT)}")
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 3)


def sample_stdev(values: list[float]) -> float:
    return round(statistics.stdev(values), 3) if len(values) > 1 else 0.0


def median(values: list[float]) -> float:
    return round(statistics.median(values), 3)


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate percentile of empty values")
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + fraction * (ordered[upper] - ordered[lower]), 3)


def aggregate(benchmark: str, summary_path: Path, summary: dict[str, Any], mode: str) -> dict[str, Any]:
    runs = [row for row in summary["runs"] if row["mode"] == mode]
    passed = [row for row in runs if row["passed"]]
    tokens = [float(row["total_tokens"]) for row in runs]
    latencies = [float(row["total_api_latency_seconds"]) for row in runs]
    iterations = [float(row["iterations_executed"]) for row in runs]
    success_iterations = [str(row["success_iteration"]) for row in passed]
    distribution: dict[str, int] = {}
    for value in success_iterations:
        distribution[value] = distribution.get(value, 0) + 1
    return {
        "benchmark": benchmark,
        "mode": mode,
        "runs": len(runs),
        "passed_runs": len(passed),
        "success_rate": round(len(passed) / len(runs), 4),
        "mean_tokens": mean(tokens),
        "stdev_tokens": sample_stdev(tokens),
        "mean_latency_seconds": mean(latencies),
        "stdev_latency_seconds": sample_stdev(latencies),
        "median_latency_seconds": median(latencies),
        "p90_latency_seconds": percentile(latencies, 0.90),
        "mean_iterations": mean(iterations),
        "success_iteration_distribution": distribution,
        "source_summary": str(summary_path.relative_to(ROOT)),
    }


def combined(rows: list[dict[str, Any]], summaries: dict[str, dict[str, Any]], mode: str) -> dict[str, Any]:
    all_runs: list[dict[str, Any]] = []
    for summary in summaries.values():
        all_runs.extend(row for row in summary["runs"] if row["mode"] == mode)
    passed = [row for row in all_runs if row["passed"]]
    tokens = [float(row["total_tokens"]) for row in all_runs]
    latencies = [float(row["total_api_latency_seconds"]) for row in all_runs]
    iterations = [float(row["iterations_executed"]) for row in all_runs]
    distribution: dict[str, int] = {}
    for row in passed:
        key = str(row["success_iteration"])
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "benchmark": "COMBINED",
        "mode": mode,
        "runs": len(all_runs),
        "passed_runs": len(passed),
        "success_rate": round(len(passed) / len(all_runs), 4),
        "mean_tokens": mean(tokens),
        "stdev_tokens": sample_stdev(tokens),
        "mean_latency_seconds": mean(latencies),
        "stdev_latency_seconds": sample_stdev(latencies),
        "median_latency_seconds": median(latencies),
        "p90_latency_seconds": percentile(latencies, 0.90),
        "mean_iterations": mean(iterations),
        "success_iteration_distribution": distribution,
        "source_summary": "ATAX+BICG latest repeated summaries",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {
        name: latest_summary(path) for name, path in BENCHMARKS.items()
    }
    summaries = {name: value[1] for name, value in loaded.items()}

    rows = [
        aggregate(name, summary_path, summary, mode)
        for name, (summary_path, summary) in loaded.items()
        for mode in ("one_shot", "iterative")
    ]
    rows.extend(combined(rows, summaries, mode) for mode in ("one_shot", "iterative"))

    one = next(row for row in rows if row["benchmark"] == "COMBINED" and row["mode"] == "one_shot")
    iterative = next(row for row in rows if row["benchmark"] == "COMBINED" and row["mode"] == "iterative")
    token_overhead = round(
        (iterative["mean_tokens"] - one["mean_tokens"]) / one["mean_tokens"], 4
    )
    median_latency_overhead = round(
        (iterative["median_latency_seconds"] - one["median_latency_seconds"])
        / one["median_latency_seconds"],
        4,
    )
    effect = {
        "success_rate_change_percentage_points": round(
            100 * (iterative["success_rate"] - one["success_rate"]), 1
        ),
        "relative_token_overhead": token_overhead,
        "relative_median_latency_overhead": median_latency_overhead,
        "all_iterative_successes_on_iteration_2": (
            iterative["success_iteration_distribution"] == {"2": iterative["passed_runs"]}
        ),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "results/comparisons/atax_vs_bicg_staged_feedback" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "benchmark_comparison.csv", rows)
    (output_dir / "comparison.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "comparison": "atax_vs_bicg_staged_feedback",
                "timestamp_utc": timestamp,
                "rows": rows,
                "combined_effect": effect,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("benchmark | mode | passed/runs | success | mean tokens ± sd | median latency | p90 latency")
    print("--- | --- | --- | --- | --- | --- | ---")
    for row in rows:
        print(
            f"{row['benchmark']} | {row['mode']} | "
            f"{row['passed_runs']}/{row['runs']} | {100 * row['success_rate']:.1f}% | "
            f"{row['mean_tokens']} ± {row['stdev_tokens']} | "
            f"{row['median_latency_seconds']} | {row['p90_latency_seconds']}"
        )
    print("\nCombined effect")
    print(json.dumps(effect, indent=2))
    print(f"Results: {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
