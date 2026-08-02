#!/usr/bin/env python3

"""Compare latest repeated staged-repair summaries across configured models."""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    ("Qwen3.5-122B-A10B", "qwen35", None),
    ("DeepSeek-V4-Pro", "deepseek_v4_pro", "staged_deepseek_v4_pro_repeated"),
    ("Kimi-K2.7-Code", "kimi_k27_code", "staged_kimi_k27_code_repeated"),
]


def latest_summary(model_slug: str, directory_name: str | None) -> Path:
    if model_slug == "qwen35":
        raise ValueError("Qwen summaries are benchmark-specific")
    root = ROOT / "results/ablations" / str(directory_name)
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and (p / "summary.json").is_file())
    if not candidates:
        raise RuntimeError(f"No completed summary found under {root}")
    return candidates[-1] / "summary.json"


def latest_qwen_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, dirname in (
        ("BICG", "bicg_staged_feedback_repeated"),
        ("ATAX", "atax_staged_feedback_repeated"),
    ):
        root = ROOT / "results/ablations" / dirname
        candidates = sorted(p for p in root.iterdir() if p.is_dir() and (p / "summary.json").is_file())
        if not candidates:
            raise RuntimeError(f"No completed Qwen summary under {root}")
        summary_path = candidates[-1] / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary["runs"]:
            copied = dict(row)
            copied["benchmark"] = benchmark
            copied["source_summary"] = str(summary_path.relative_to(ROOT))
            rows.append(copied)
    return rows


def model_runs(model_slug: str, directory_name: str | None) -> list[dict[str, Any]]:
    if model_slug == "qwen35":
        return latest_qwen_runs()
    summary_path = latest_summary(model_slug, directory_name)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for row in summary["runs"]:
        copied = dict(row)
        copied["source_summary"] = str(summary_path.relative_to(ROOT))
        rows.append(copied)
    return rows


def nums(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def aggregate(model: str, rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row["mode"] == mode]
    passed = [row for row in selected if row["passed"]]
    tokens = nums(selected, "total_tokens")
    latency = nums(selected, "total_api_latency_seconds")
    iterations = nums(selected, "iterations_executed")
    success_iterations: dict[str, int] = {}
    for row in passed:
        key = str(row.get("success_iteration"))
        success_iterations[key] = success_iterations.get(key, 0) + 1
    return {
        "model": model,
        "mode": mode,
        "runs": len(selected),
        "passed_runs": len(passed),
        "success_rate": round(len(passed) / len(selected), 4),
        "mean_tokens": round(statistics.mean(tokens), 3),
        "stdev_tokens": round(statistics.stdev(tokens), 3) if len(tokens) > 1 else 0.0,
        "mean_latency_seconds": round(statistics.mean(latency), 3),
        "stdev_latency_seconds": round(statistics.stdev(latency), 3) if len(latency) > 1 else 0.0,
        "median_latency_seconds": round(statistics.median(latency), 3),
        "mean_iterations": round(statistics.mean(iterations), 3),
        "success_iteration_distribution": success_iterations,
    }


def main() -> None:
    all_rows: dict[str, list[dict[str, Any]]] = {}
    output_rows: list[dict[str, Any]] = []
    for model_name, slug, dirname in MODELS:
        rows = model_runs(slug, dirname)
        all_rows[model_name] = rows
        output_rows.append(aggregate(model_name, rows, "one_shot"))
        output_rows.append(aggregate(model_name, rows, "iterative"))

    qwen_iter = next(r for r in output_rows if r["model"].startswith("Qwen") and r["mode"] == "iterative")
    rankings = sorted(
        (r for r in output_rows if r["mode"] == "iterative"),
        key=lambda r: (-r["success_rate"], r["mean_tokens"], r["median_latency_seconds"]),
    )
    for rank, row in enumerate(rankings, start=1):
        row["efficiency_rank"] = rank
        row["token_overhead_vs_qwen"] = round(row["mean_tokens"] / qwen_iter["mean_tokens"] - 1.0, 4)
        row["median_latency_overhead_vs_qwen"] = round(
            row["median_latency_seconds"] / qwen_iter["median_latency_seconds"] - 1.0, 4
        )
    for row in output_rows:
        row.setdefault("efficiency_rank", None)
        row.setdefault("token_overhead_vs_qwen", None)
        row.setdefault("median_latency_overhead_vs_qwen", None)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results/comparisons/staged_three_model" / timestamp
    out.mkdir(parents=True, exist_ok=False)

    with (out / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "schema_version": 1,
        "comparison": "staged_three_model",
        "timestamp_utc": timestamp,
        "rows": output_rows,
        "recommended_default_model": rankings[0]["model"],
        "selection_rule": "Highest iterative success, then lowest mean tokens, then lowest median latency",
        "notable_behaviour": {
            "kimi_one_shot_successes": sum(
                1 for row in all_rows["Kimi-K2.7-Code"] if row["mode"] == "one_shot" and row["passed"]
            )
        },
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Three-model staged repair comparison")
    print("model | mode | passed/runs | success | mean tokens | median latency | rank")
    print("--- | --- | --- | --- | --- | --- | ---")
    for row in output_rows:
        rank = row["efficiency_rank"] if row["efficiency_rank"] is not None else "-"
        print(
            f"{row['model']} | {row['mode']} | {row['passed_runs']}/{row['runs']} | "
            f"{100 * row['success_rate']:.1f}% | {row['mean_tokens']} | "
            f"{row['median_latency_seconds']} | {rank}"
        )
    print(f"Recommended default: {rankings[0]['model']}")
    print(f"Results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
