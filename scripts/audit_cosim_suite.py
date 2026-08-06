#!/usr/bin/env python3

"""Re-audit final/Pareto designs for every run in a materialised suite.

This command is intentionally separate from the metered autonomous-search
budget. It reuses durable candidate evidence, but it never trusts a legacy
baseline boolean unless ``verified_baseline.json`` or a previous non-reused
final audit proves that C/RTL co-simulation actually ran.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.config import load_task  # noqa: E402
from agent.final_cosim import enforce_final_cosim_policy  # noqa: E402
from agent.state import AgentResult, TrajectoryEvent  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _result(payload: dict[str, Any], output_dir: Path) -> AgentResult:
    trajectory = []
    for item in payload.get("trajectory", []):
        if not isinstance(item, dict):
            continue
        trajectory.append(
            TrajectoryEvent(
                step=int(item.get("step", len(trajectory) + 1)),
                stage=str(item.get("stage", "unknown")),
                status=str(item.get("status", "unknown")),
                details=(
                    dict(item.get("details") or {})
                    if isinstance(item.get("details"), dict)
                    else {}
                ),
            )
        )
    return AgentResult(
        task_id=str(payload.get("task_id", output_dir.name)),
        success=payload.get("success") is True,
        status=str(payload.get("status", "unknown")),
        termination_reason=str(payload.get("termination_reason", "unknown")),
        output_dir=str(output_dir),
        trajectory=trajectory,
    )


def _legacy_baseline_was_reused(output_dir: Path) -> bool:
    audit = _load(output_dir / "final_cosim_audit.json")
    return any(
        isinstance(item, dict)
        and item.get("candidate_index") == 0
        and item.get("reused_existing_cosim") is True
        for item in audit.get("candidates", [])
    )


def _durable_baseline_cosim(output_dir: Path) -> bool | None:
    baseline = _load(output_dir / "verified_baseline.json")
    validation = baseline.get("validation")
    validation = validation if isinstance(validation, dict) else {}
    value = validation.get("cosim_passed")
    return value if isinstance(value, bool) else None


def _clear_record_cosim(record: Any) -> Any:
    if not isinstance(record, dict) or record.get("candidate_index") != 0:
        return record
    record = dict(record)
    validation = record.get("validation")
    validation = dict(validation) if isinstance(validation, dict) else {}
    validation["cosim"] = None
    record["validation"] = validation
    record["cosim"] = None
    record["cosim_run"] = False
    record["fully_verified"] = False
    record.pop("final_cosim", None)
    return record


def _sanitise_legacy_baseline(output_dir: Path) -> bool:
    state_path = output_dir / "candidate_state.json"
    state = _load(state_path)
    if not state:
        return False

    durable = _durable_baseline_cosim(output_dir)
    suspicious = _legacy_baseline_was_reused(output_dir)
    if durable is True and not suspicious:
        return False

    changed = False
    for key in (
        "selected_design",
        "latest_candidate",
        "best_correct_candidate",
        "best_ppa_candidate",
        "original_baseline",
    ):
        original = state.get(key)
        replacement = _clear_record_cosim(original)
        if replacement != original:
            state[key] = replacement
            changed = True

    pareto = []
    for item in state.get("pareto_archive", []):
        replacement = _clear_record_cosim(item)
        changed = changed or replacement != item
        pareto.append(replacement)
    state["pareto_archive"] = pareto

    if changed:
        state["selected_design_fully_verified"] = bool(
            isinstance(state.get("selected_design"), dict)
            and state["selected_design"].get("fully_verified") is True
        )
        _write(state_path, state)
    return changed


def _normalise_verified_budget_stop(result: AgentResult, audit: dict[str, Any]) -> None:
    if (
        audit.get("status") in {"passed", "passed_with_fallback"}
        and result.termination_reason == "final_verification_budget_unavailable"
    ):
        result.success = True
        if audit.get("fallback_used") is True:
            result.status = "completed_with_cosim_fallback"
            result.termination_reason = "selected_cosim_failed_fallback_verified"
        else:
            result.status = "completed_budget"
            result.termination_reason = "verified_result_selected_after_post_search_cosim"


def _selected_index(output_dir: Path) -> Any:
    state = _load(output_dir / "candidate_state.json")
    selected = state.get("selected_design")
    return selected.get("candidate_index") if isinstance(selected, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run durable final/Pareto co-simulation audits for a suite."
    )
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_dir = REPO_ROOT / "results" / "suites" / args.run_id
    matrix = _load(run_dir / "matrix_manifest.json")
    runs = matrix.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit(f"No materialised suite found under {run_dir}")

    rows: list[dict[str, Any]] = []
    failures = 0
    for position, item in enumerate(runs, 1):
        if not isinstance(item, dict):
            continue
        run_key = str(item.get("run_key", f"run_{position}"))
        output_dir = _resolve(str(item["output_dir"]))
        manifest_path = _resolve(str(item["manifest_path"]))
        result_path = output_dir / "unified_agent_result.json"
        payload = _load(result_path)

        print(f"[{position}/{len(runs)}] {run_key}", flush=True)
        if not payload:
            print("  skipped: no unified result", flush=True)
            rows.append({"run_key": run_key, "status": "skipped_no_result"})
            failures += 1
            continue

        sanitised = _sanitise_legacy_baseline(output_dir)
        task = load_task(manifest_path)
        result = _result(payload, output_dir)
        audit = enforce_final_cosim_policy(task, result)
        _normalise_verified_budget_stop(result, audit)
        _write(result_path, result.to_dict())

        status = str(audit.get("status"))
        passed = status in {"passed", "passed_with_fallback"}
        failures += 0 if passed else 1
        row = {
            "run_key": run_key,
            "task_id": item.get("task_id"),
            "model_slug": item.get("model_slug"),
            "audit_status": status,
            "fallback_used": audit.get("fallback_used"),
            "selected_candidate_index": _selected_index(output_dir),
            "verified_pareto_count": audit.get("verified_pareto_count"),
            "candidates_audited": len(audit.get("candidates", [])),
            "legacy_baseline_sanitised": sanitised,
            "success": result.success,
            "status": result.status,
            "termination_reason": result.termination_reason,
        }
        rows.append(row)
        print(
            "  audit="
            f"{status} selected={row['selected_candidate_index']} "
            f"pareto={row['verified_pareto_count']} "
            f"fallback={row['fallback_used']} "
            f"baseline_sanitised={sanitised}",
            flush=True,
        )

    json_path = run_dir / "final_cosim_suite_summary.json"
    csv_path = run_dir / "final_cosim_suite_summary.csv"
    _write(
        json_path,
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "total_runs": len(rows),
            "failed_audits": failures,
            "results": rows,
        },
    )
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Summary: {json_path.relative_to(REPO_ROOT)}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
