#!/usr/bin/env python3

"""Re-materialise manifests for an existing matrix without deleting results."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = REPO_ROOT / "scripts" / "run_experiment_matrix.py"
STALE_STAGE_AWARE_TASKS = {
    "vector_add_generate",
    "synth_fix_dynamic_buffer",
    "residual_stream_deadlock",
    "structural_blind_stream",
}


def _matrix_module():
    name = "run_experiment_matrix_refresh"
    spec = importlib.util.spec_from_file_location(name, MATRIX_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MATRIX_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _invalidate_stale_rows(state: dict) -> int:
    rows = state.get("results")
    if not isinstance(rows, list):
        return 0

    count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if (
            row.get("task_id") in STALE_STAGE_AWARE_TASKS
            and row.get("success") is True
        ):
            row["success"] = None
            row["status"] = "stale_dispatch"
            row["termination_reason"] = "rerun_required_after_manifest_dispatch_fix"
            row["final_design_verified"] = None
            row["error"] = (
                "This row used generic repair because its materialised JSON manifest "
                "was not loaded before stage-aware workflow dispatch."
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh generated task manifests for an existing experiment matrix "
            "while preserving suite state, logs and completed outputs."
        )
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    state_path = run_dir / "suite_state.json"
    suite_path = run_dir / "suite_definition.json"
    if not state_path.is_file() or not suite_path.is_file():
        raise RuntimeError(
            f"Expected suite_state.json and suite_definition.json under {run_dir}"
        )

    state = _load_json(state_path)
    run_id = str(state.get("run_id", "")).strip()
    total_runs = int(state.get("total_runs", 0))
    if not run_id or total_runs <= 0:
        raise RuntimeError("Existing suite state has no valid run_id/total_runs")

    module = _matrix_module()
    suite = module._load_suite(suite_path)
    runs = module.materialise_runs(
        suite,
        run_id=run_id,
        run_dir=run_dir,
        core_only=False,
        maximum=total_runs,
    )
    if len(runs) != total_runs:
        raise RuntimeError(
            f"Refreshed {len(runs)} manifests but suite state expects {total_runs}"
        )

    invalidated = _invalidate_stale_rows(state)
    state["status"] = "recovery_ready"
    state["finished_at"] = None
    state["recovery"] = {
        "manifests_refreshed": True,
        "stage_aware_rows_invalidated": invalidated,
        "stale_task_ids": sorted(STALE_STAGE_AWARE_TASKS),
    }
    module._write_json_atomic(state_path, state)
    rows = [row for row in state.get("results", []) if isinstance(row, dict)]
    module._write_csv(run_dir / "suite_summary.csv", rows)

    print(f"Refreshed {len(runs)} manifests under {run_dir / 'manifests'}")
    print(f"Marked {invalidated} stale successful stage-aware row(s) for rerun.")
    print("Existing logs, completed outputs and valid result rows were preserved.")


if __name__ == "__main__":
    main()
