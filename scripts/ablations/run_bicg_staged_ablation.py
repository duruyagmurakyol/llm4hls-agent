#!/usr/bin/env python3

"""Run one-shot versus three-iteration staged BICG repair ablation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/experiments/run_iterative_api_experiment.py"
CASES = [
    "staged_compile_then_functional",
    "staged_interface_then_functional",
    "staged_compile_compile_functional",
]


def latest_result(experiment_id: str) -> Path:
    root = ROOT / "runs/experiments" / experiment_id
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise RuntimeError(f"No results found for {experiment_id}")
    return candidates[-1] / "result.json"


def run_case(case: str, max_iterations: int) -> dict[str, object]:
    config = ROOT / "configs/bicg_iterative_qwen35" / f"{case}.json"
    command = [
        sys.executable,
        str(RUNNER),
        str(config),
        "--max-iterations",
        str(max_iterations),
        "--keep-workspace",
    ]
    process = subprocess.run(command, cwd=ROOT, text=True, check=False)

    experiment_id = f"bicg_iterative_qwen35_{case}"
    result_path = latest_result(experiment_id)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "case": case,
        "mode": "one_shot" if max_iterations == 1 else "iterative",
        "max_iterations": max_iterations,
        "runner_return_code": process.returncode,
        "passed": bool(result.get("passed", False)),
        "iterations_executed": result.get("iterations_executed"),
        "success_iteration": result.get("success_iteration"),
        "total_tokens": result.get("total_tokens"),
        "total_api_latency_seconds": result.get("total_api_latency_seconds"),
        "post_host_validation_passed": result.get("post_host_validation_passed"),
        "independent_validation_passed": result.get("independent_validation_passed"),
        "result_path": str(result_path.relative_to(ROOT)),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for case in CASES:
        rows.append(run_case(case, 1))
        rows.append(run_case(case, 3))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "results/ablations/bicg_staged_feedback" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_version": 1,
        "ablation": "bicg_staged_feedback",
        "timestamp_utc": timestamp,
        "model": "Qwen/Qwen3.5-122B-A10B",
        "cases": len(CASES),
        "runs": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fields = list(rows[0].keys())
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nAblation summary")
    print("case | one-shot | iterative | success iteration | one-shot tokens | iterative tokens")
    print("--- | --- | --- | --- | --- | ---")
    for case in CASES:
        one = next(row for row in rows if row["case"] == case and row["mode"] == "one_shot")
        iterative = next(row for row in rows if row["case"] == case and row["mode"] == "iterative")
        print(
            f"{case} | {one['passed']} | {iterative['passed']} | "
            f"{iterative['success_iteration']} | {one['total_tokens']} | "
            f"{iterative['total_tokens']}"
        )
    print(f"Results: {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
