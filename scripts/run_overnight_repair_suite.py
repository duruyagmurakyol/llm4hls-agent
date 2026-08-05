#!/usr/bin/env python3

"""Run the controlled repair suite repeatedly and retain partial summaries.

Runs are deliberately sequential because Vitis workspaces and licences are more
reliable that way. Every repetition receives a unique task ID, output directory
and repair experiment ID, so no candidate or validation evidence is inherited.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = REPO_ROOT / "configs" / "tasks" / "repair_suite" / "index.json"
RUN_AGENT = REPO_ROOT / "scripts" / "run_agent.py"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def safe_fragment(value: str) -> str:
    result = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    result = result.strip("._-")
    if not result:
        raise ValueError(f"Could not form a safe identifier from {value!r}")
    return result


def task_paths(index_path: Path, selected_cases: set[str] | None) -> list[Path]:
    index = load_json(index_path)
    raw = index.get("cases")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"No repair cases declared in {index_path}")

    paths: list[Path] = []
    for item in raw:
        path = resolve(str(item))
        case_name = path.stem
        if selected_cases and case_name not in selected_cases:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Repair task manifest not found: {path}")
        paths.append(path)

    if selected_cases:
        found = {path.stem for path in paths}
        missing = sorted(selected_cases - found)
        if missing:
            raise ValueError("Unknown --case value(s): " + ", ".join(missing))
    return paths


def validate_task_inputs(task_path: Path) -> list[str]:
    task = load_json(task_path)
    errors: list[str] = []
    for key in ("task_id", "artifacts", "interface", "target", "budgets", "model", "repair", "output_dir"):
        if key not in task:
            errors.append(f"missing task.{key}")

    artifacts = task.get("artifacts") if isinstance(task.get("artifacts"), dict) else {}
    values: list[tuple[str, Any]] = [
        ("source", artifacts.get("source")),
        ("build_file", (artifacts.get("build_files") or [None])[0]),
        ("testbench", (artifacts.get("testbench") or [None])[0]),
        ("header", (artifacts.get("headers") or [None])[0]),
    ]
    for label, value in values:
        if not isinstance(value, str) or not resolve(value).is_file():
            errors.append(f"missing {label}: {value!r}")

    repair = task.get("repair") if isinstance(task.get("repair"), dict) else {}
    benchmark_source = repair.get("benchmark_source")
    if not isinstance(benchmark_source, str) or not resolve(benchmark_source).is_dir():
        errors.append(f"missing repair benchmark source: {benchmark_source!r}")

    if task.get("adapter", {}).get("kind") != "direct_api_repair":
        errors.append("adapter.kind must be direct_api_repair")
    return errors


def derived_manifest(
    source_path: Path,
    *,
    run_id: str,
    repetition: int,
    suite_root: Path,
) -> tuple[Path, dict[str, Any]]:
    source = load_json(source_path)
    base_id = safe_fragment(str(source["task_id"]))
    unique_id = f"{base_id}__{run_id}__r{repetition:02d}"
    task = json.loads(json.dumps(source))
    task["task_id"] = unique_id
    task["parent_task_id"] = base_id
    task["suite_run_id"] = run_id
    task["suite_repetition"] = repetition
    task["output_dir"] = relative(suite_root / "tasks" / unique_id)

    manifest_path = suite_root / "manifests" / f"{unique_id}.json"
    write_json(manifest_path, task)
    return manifest_path, task


def run_process(command: list[str], *, log_path: Path, timeout_seconds: int) -> tuple[int | None, bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    return_code: int | None = None

    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            log.write(f"\nTIMEOUT after {timeout_seconds} seconds; terminating process group.\n")
            log.flush()
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            return_code = process.returncode

    return return_code, timed_out, time.monotonic() - started


def optional_json(path: Path) -> dict[str, Any]:
    try:
        return load_json(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {}


def latest_repair_result(task_id: str) -> tuple[Path | None, dict[str, Any]]:
    roots = [
        REPO_ROOT / "results" / "experiments" / task_id,
        REPO_ROOT / "runs" / "experiments" / task_id,
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.is_dir():
            candidates.extend(root.glob("*/result.json"))
    if not candidates:
        return None, {}
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    return path, optional_json(path)


def trajectory_stage(result: dict[str, Any], stage: str) -> str | None:
    values = [
        str(item.get("status"))
        for item in result.get("trajectory", [])
        if isinstance(item, dict) and item.get("stage") == stage
    ]
    return values[-1] if values else None


def make_row(
    *,
    source_task: Path,
    task: dict[str, Any],
    repetition: int,
    return_code: int | None,
    timed_out: bool,
    elapsed_seconds: float,
    log_path: Path,
) -> dict[str, Any]:
    output_dir = resolve(str(task["output_dir"]))
    unified = optional_json(output_dir / "unified_agent_result.json")
    budget = optional_json(output_dir / "budget_summary.json")
    repair_path, repair = latest_repair_result(str(task["task_id"]))

    parent_id = str(task.get("parent_task_id", source_task.stem))
    benchmark, _, case = parent_id.partition("_")
    if parent_id.startswith("dot_product_"):
        benchmark = "dot_product"
        case = parent_id.removeprefix("dot_product_").removesuffix("_repair")
    elif parent_id.startswith("gemm_"):
        benchmark = "gemm"
        case = parent_id.removeprefix("gemm_").removesuffix("_repair")

    modified = repair.get("modified_files")
    if isinstance(modified, list):
        modified_text = ";".join(str(item) for item in modified)
    else:
        modified_text = ""

    return {
        "benchmark": benchmark,
        "case": case,
        "repetition": repetition,
        "task_id": task["task_id"],
        "source_manifest": relative(source_task),
        "derived_manifest": relative(resolve(str(task["output_dir"])).parents[1] / "manifests" / f"{task['task_id']}.json"),
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "success": unified.get("success", repair.get("passed")),
        "status": unified.get("status"),
        "termination_reason": unified.get("termination_reason"),
        "initial_failure_class": repair.get("failure_class"),
        "repair_attempt_count": repair.get("attempt_count", repair.get("attempt")),
        "pre_host_validation_passed": repair.get("pre_host_validation_passed"),
        "post_host_validation_passed": repair.get("post_host_validation_passed"),
        "independent_validation_passed": repair.get("independent_validation_passed"),
        "post_repair_synthesis": trajectory_stage(unified, "post_repair_synthesis"),
        "post_repair_cosim": trajectory_stage(unified, "post_repair_cosim"),
        "input_tokens": repair.get("input_tokens"),
        "output_tokens": repair.get("output_tokens"),
        "total_tokens": repair.get("tokens_used"),
        "model": repair.get("model", task.get("model", {}).get("name")),
        "modified_files": modified_text,
        "changed_line_count": repair.get("changed_line_count"),
        "budget_iterations_used": budget.get("iterations_used"),
        "budget_model_calls_used": budget.get("model_calls_used"),
        "budget_csim_calls_used": budget.get("csim_calls_used"),
        "budget_synthesis_calls_used": budget.get("synthesis_calls_used"),
        "budget_cosim_calls_used": budget.get("cosim_calls_used"),
        "log_file": relative(log_path),
        "repair_result": relative(repair_path) if repair_path else None,
        "unified_result": relative(output_dir / "unified_agent_result.json"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], *, run_id: str, expected: int) -> dict[str, Any]:
    completed = len(rows)
    successes = sum(row.get("success") is True for row in rows)
    timeouts = sum(row.get("timed_out") is True for row in rows)
    total_tokens = sum(
        int(value)
        for row in rows
        if isinstance((value := row.get("total_tokens")), int)
    )
    by_benchmark: dict[str, dict[str, int]] = {}
    for row in rows:
        name = str(row["benchmark"])
        item = by_benchmark.setdefault(name, {"runs": 0, "successes": 0})
        item["runs"] += 1
        item["successes"] += int(row.get("success") is True)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "expected_runs": expected,
        "completed_runs": completed,
        "successful_runs": successes,
        "failed_runs": completed - successes,
        "timed_out_runs": timeouts,
        "total_tokens": total_tokens,
        "by_benchmark": by_benchmark,
        "rows": rows,
    }


def preflight(index_path: Path, tasks: list[Path]) -> None:
    problems: list[str] = []
    if not RUN_AGENT.is_file():
        problems.append(f"missing runner: {RUN_AGENT}")
    if not os.environ.get("SILICONFLOW_API_KEY") and not os.environ.get("SILICONFLOW_KEY"):
        problems.append("SILICONFLOW_API_KEY (or SILICONFLOW_KEY) is not set")
    if not any(
        Path(directory, "vitis-run").is_file()
        for directory in os.environ.get("PATH", "").split(os.pathsep)
        if directory
    ):
        problems.append("vitis-run is not available in PATH")

    for task in tasks:
        for error in validate_task_inputs(task):
            problems.append(f"{relative(task)}: {error}")

    if problems:
        raise RuntimeError(
            "Overnight-suite preflight failed:\n- " + "\n- ".join(problems)
        )
    print(f"Preflight passed: {len(tasks)} cases from {relative(index_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated controlled HLS repair experiments sequentially."
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.timeout_seconds < 60:
        raise ValueError("--timeout-seconds must be at least 60")

    index_path = args.index.expanduser().resolve()
    selected = {safe_fragment(item) for item in args.case} or None
    tasks = task_paths(index_path, selected)
    preflight(index_path, tasks)

    expected = len(tasks) * args.repeats
    if args.dry_run:
        print(f"Planned runs: {len(tasks)} cases x {args.repeats} repetitions = {expected}")
        for task in tasks:
            print("-", relative(task))
        return

    run_id = safe_fragment(args.run_id or f"repair_{utc_stamp()}")
    suite_root = REPO_ROOT / "runs" / "overnight_repair" / run_id
    if suite_root.exists() and any(suite_root.iterdir()):
        raise FileExistsError(
            f"Suite output already exists: {suite_root}. Choose a new --run-id."
        )
    suite_root.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    summary_path = suite_root / "summary.json"
    csv_path = suite_root / "summary.csv"
    write_json(
        suite_root / "plan.json",
        {
            "run_id": run_id,
            "index": relative(index_path),
            "cases": [relative(path) for path in tasks],
            "repeats": args.repeats,
            "timeout_seconds": args.timeout_seconds,
            "expected_runs": expected,
        },
    )

    sequence = 0
    for repetition in range(1, args.repeats + 1):
        for source_task in tasks:
            sequence += 1
            manifest_path, task = derived_manifest(
                source_task,
                run_id=run_id,
                repetition=repetition,
                suite_root=suite_root,
            )
            log_path = suite_root / "logs" / f"{task['task_id']}.log"
            print(
                f"[{sequence}/{expected}] {source_task.stem}, repetition {repetition}",
                flush=True,
            )
            return_code, timed_out, elapsed = run_process(
                [sys.executable, "-u", str(RUN_AGENT), str(manifest_path)],
                log_path=log_path,
                timeout_seconds=args.timeout_seconds,
            )
            row = make_row(
                source_task=source_task,
                task=task,
                repetition=repetition,
                return_code=return_code,
                timed_out=timed_out,
                elapsed_seconds=elapsed,
                log_path=log_path,
            )
            rows.append(row)
            write_json(summary_path, aggregate(rows, run_id=run_id, expected=expected))
            write_csv(csv_path, rows)
            print(
                f"    success={row['success']} status={row['status']} "
                f"tokens={row['total_tokens']} elapsed={row['elapsed_seconds']}s",
                flush=True,
            )

    summary = aggregate(rows, run_id=run_id, expected=expected)
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(summary_path, summary)
    print("\nOvernight repair suite complete")
    print("Summary:", relative(summary_path))
    print("CSV:", relative(csv_path))
    print(f"Successful runs: {summary['successful_runs']}/{summary['completed_runs']}")


if __name__ == "__main__":
    main()
