#!/usr/bin/env python3

"""Run one FPT 2026 Track A task through a budgeted autonomous state machine."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def run_python(arguments: list[str], title: str) -> int:
    if arguments and arguments[0] == "-m":
        command = [sys.executable, *arguments]
    else:
        command = [
            sys.executable,
            *[str(resolve(item)) if i == 0 else item for i, item in enumerate(arguments)],
        ]
    print(f"\n=== {title} ===", flush=True)
    print("Command:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_usage(output_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    totals = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for path in sorted(output_dir.glob("candidate_*_model_metadata.json")):
        data = load_json(path)
        records.append({"file": str(path.relative_to(REPO_ROOT)), **data})
        totals["model_calls"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if isinstance(data.get(key), int):
                totals[key] += data[key]
    return {"totals": totals, "calls": records}


def derive_phase(summary: dict[str, Any], task: dict[str, Any]) -> str:
    candidates = summary.get("candidates", [])
    if not candidates:
        return "diagnose_initial_design"
    verdict = candidates[-1].get("verdict")
    if verdict in {"reject_static", "reject_csim", "reject_csim_timeout", "reject_duplicate"}:
        return "repair_correctness"
    budget = summary.get("budget", {})
    if budget.get("synthesis_calls_remaining", 0) <= 0:
        return "terminated_budget"
    if len(candidates) >= task["budgets"]["max_iterations"]:
        return "terminated_iteration_limit"
    if verdict in {
        "keep_pareto_candidate",
        "accept_dominates_baseline",
        "reject_no_change",
        "reject_no_performance_gain",
        "reject_synthesis_timeout",
        "reject_synthesis_failed",
    }:
        return "optimise_ppa"
    return "continue_evaluation"


def write_ledger(
    task_path: Path,
    task: dict[str, Any],
    summary: dict[str, Any] | None,
    status: str,
) -> Path:
    output_dir = resolve(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_config = load_json(resolve(task["adapter"]["config"]))
    adapter_output = resolve(adapter_config["output_dir"])
    usage = model_usage(adapter_output)
    tool_usage = {
        "csim_calls": sum(
            1
            for path in adapter_output.glob("candidate_*_csim_validation.json")
            if load_json(path).get("passed") is not None
        ),
        "cosim_calls": sum(1 for _ in adapter_output.glob("candidate_*_cosim_validation.json")),
        "synthesis_calls": (summary or {}).get("budget", {}).get("synthesis_calls_used", 0),
    }
    budgets = task["budgets"]
    ledger = {
        "task_id": task["task_id"],
        "manifest": (
            str(task_path.relative_to(REPO_ROOT))
            if task_path.is_relative_to(REPO_ROOT)
            else str(task_path)
        ),
        "status": status,
        "updated_at": utc_now(),
        "initial_condition": task["task_kind"],
        "policy": {
            "correctness_before_ppa": True,
            "duplicate_rejection_before_tools": True,
            "pareto_archive": True,
            "budgeted_termination": True,
            "tool_timeouts": True,
        },
        "budgets": budgets,
        "usage": {**tool_usage, **usage["totals"]},
        "remaining": {
            "csim_calls": max(0, budgets["max_csim_calls"] - tool_usage["csim_calls"]),
            "cosim_calls": max(0, budgets["max_cosim_calls"] - tool_usage["cosim_calls"]),
            "synthesis_calls": max(
                0,
                budgets["max_synthesis_calls"] - tool_usage["synthesis_calls"],
            ),
            "model_calls": max(0, budgets["max_model_calls"] - usage["totals"]["model_calls"]),
        },
        "experiment_summary": summary,
        "model_call_records": usage["calls"],
    }
    path = output_dir / "run_ledger.json"
    path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a budgeted FPT Track A LLM4HLS task.")
    parser.add_argument("task", type=Path)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--max-agent-steps", type=int, default=None)
    args = parser.parse_args()

    task_path = args.task.resolve()
    task = load_json(task_path)
    if task.get("adapter", {}).get("kind") != "legacy_ppa":
        raise ValueError("This controller currently requires a legacy_ppa adapter.")
    if run_python(["scripts/validate_track_a_task.py", str(task_path)], "Validate task package") != 0:
        raise SystemExit(1)

    adapter = task["adapter"]
    adapter_config_path = resolve(adapter["config"])
    summary_path = resolve(adapter["summary"])
    if not summary_path.is_file() and not args.status_only:
        if run_python(adapter["initialise_command"], "Diagnose and initialise baseline") != 0:
            write_ledger(task_path, task, None, "failed_initialisation")
            raise SystemExit(1)
        if (
            run_python(
                ["-m", "agent.optimise.evaluate", str(adapter_config_path)],
                "Create initial experiment summary",
            )
            != 0
        ):
            write_ledger(task_path, task, None, "failed_summary_initialisation")
            raise SystemExit(1)
        if not summary_path.is_file():
            write_ledger(task_path, task, None, "failed_summary_initialisation")
            raise FileNotFoundError(
                f"Initialisation completed but did not create summary: {summary_path}"
            )

    summary = load_json(summary_path) if summary_path.is_file() else None
    if args.status_only:
        ledger = write_ledger(task_path, task, summary, "status_only")
        print(f"\nLedger: {ledger.relative_to(REPO_ROOT)}")
        return

    step_limit = args.max_agent_steps or task["budgets"]["max_iterations"]
    previous_fingerprint: str | None = None
    final_status = "terminated_no_progress"
    for step in range(1, step_limit + 1):
        summary = load_json(summary_path)
        phase = derive_phase(summary, task)
        fingerprint = json.dumps(
            {
                "phase": phase,
                "candidate_count": len(summary.get("candidates", [])),
                "budget": summary.get("budget", {}),
                "latest": (
                    summary.get("candidates", [])[-1]
                    if summary.get("candidates")
                    else None
                ),
            },
            sort_keys=True,
        )
        print(f"\nTrack A step {step}: {phase}")
        if phase.startswith("terminated_"):
            final_status = phase
            break
        if fingerprint == previous_fingerprint:
            final_status = "terminated_no_progress"
            break
        previous_fingerprint = fingerprint
        if run_python(adapter["iteration_command"], "Autonomous repair/optimisation iteration") != 0:
            final_status = "iteration_failed"
            break
        updated = load_json(summary_path)
        if json.dumps(updated, sort_keys=True) == json.dumps(summary, sort_keys=True):
            final_status = "terminated_no_progress"
            break
        final_status = derive_phase(updated, task)

    summary = load_json(summary_path)
    ledger = write_ledger(task_path, task, summary, final_status)
    print("\n=== Track A run complete ===")
    print(f"Status: {final_status}")
    print(f"Ledger: {ledger.relative_to(REPO_ROOT)}")
    print("The ledger records per-phase model tokens, tool calls, budgets, verdicts, and Pareto results.")


if __name__ == "__main__":
    main()
