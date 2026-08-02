#!/usr/bin/env python3

"""Reusable preparation, validation, and execution helpers for Track A runs.

The functions in this module contain the Track A workflow.  The scripts in
``scripts/track_a`` deliberately remain as small command-line entry points so
existing documentation and automation continue to work.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PreparedRun:
    """Paths created for an isolated Track A run."""

    task_id: str
    run_id: str
    manifest_path: Path
    adapter_config_path: Path
    candidate_dir: Path
    ledger_dir: Path


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def safe_run_id(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    if not value:
        raise ValueError("Run ID must contain at least one usable character")
    return value


def prepare_fresh_run(task_path: Path, run_id: str, *, force: bool = False) -> PreparedRun:
    """Materialise a task manifest and clean evidence directories for one run."""

    source_task_path = task_path.resolve()
    source_task = load_json(source_task_path)
    adapter = source_task.get("adapter") or {}
    if adapter.get("kind") != "legacy_ppa":
        raise ValueError("Fresh workspace preparation currently supports legacy_ppa adapters")

    run_id = safe_run_id(run_id)
    task_id = str(source_task["task_id"])
    generated_root = REPO_ROOT / "experiments" / "track_a_runs" / task_id / run_id
    generated_config_dir = generated_root / "config"
    candidate_dir = generated_root / "candidates"
    ledger_dir = generated_root / "ledger"
    manifest_path = generated_config_dir / "task.json"
    adapter_path = generated_config_dir / "ppa.json"

    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"Run already exists: {manifest_path}\n"
            "Choose a new --run-id. Existing run evidence is never overwritten."
        )

    adapter_config_path = resolve(adapter["config"])
    adapter_config = load_json(adapter_config_path)
    if candidate_dir.exists() and any(candidate_dir.iterdir()):
        raise FileExistsError(
            f"Candidate workspace is not empty: {candidate_dir}\n"
            "Choose a new --run-id; fresh runs never inherit or delete candidate evidence."
        )

    candidate_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir.mkdir(parents=True, exist_ok=True)

    fresh_adapter = json.loads(json.dumps(adapter_config))
    fresh_adapter["experiment_name"] = f"{adapter_config['experiment_name']}__{run_id}"
    fresh_adapter["output_dir"] = relative(candidate_dir)
    fresh_adapter.setdefault("baseline", {})
    fresh_adapter["baseline"]["project_dir"] = relative(candidate_dir / "baseline_project")
    budgets = source_task["budgets"]
    fresh_adapter.setdefault("budget", {})
    fresh_adapter["budget"]["max_candidates"] = int(budgets["max_iterations"])
    fresh_adapter["budget"]["max_synthesis_calls"] = int(budgets["max_synthesis_calls"])
    fresh_adapter["model"] = json.loads(json.dumps(source_task["model"]))
    write_json(adapter_path, fresh_adapter)

    fresh_task = json.loads(json.dumps(source_task))
    fresh_task.update({
        "task_id": f"{task_id}__{run_id}",
        "parent_task_id": task_id,
        "run_id": run_id,
        "output_dir": relative(ledger_dir),
        "adapter": {
            "kind": "legacy_ppa",
            "config": relative(adapter_path),
            "initialise_command": ["scripts/ppa/bootstrap_ppa_agent_run.py", relative(adapter_path)],
            "iteration_command": ["scripts/ppa/run_ppa_agent_iteration.py", relative(adapter_path), "--allow-synthesis"],
            "summary": relative(candidate_dir / "experiment_summary.json"),
        },
        "provenance": {
            "source_manifest": relative(source_task_path),
            "workspace_isolation": {
                "inherits_candidate_evidence": False, "inherits_model_responses": False,
                "inherits_validation_reports": False, "inherits_pareto_archive": False,
                "inherits_baseline_reports": False, "baseline_is_synthesised_in_run": True,
            },
        },
    })
    write_json(manifest_path, fresh_task)
    return PreparedRun(task_id, run_id, manifest_path, adapter_path, candidate_dir, ledger_dir)


def validate_task(task_path: Path) -> dict[str, Any]:
    """Validate a Track A task manifest and write its validation report."""

    task_path = task_path.resolve()
    task = load_json(task_path)
    for key in ("task_id", "task_kind", "artifacts", "interface", "target", "budgets", "model", "output_dir"):
        if key not in task:
            raise ValueError(f"Missing task.{key}")
    artifacts, interface, target, budgets = (task[key] for key in ("artifacts", "interface", "target", "budgets"))
    def required(mapping: dict[str, Any], key: str, context: str) -> Any:
        if key not in mapping:
            raise ValueError(f"Missing {context}.{key}")
        return mapping[key]
    paths = {
        "source": resolve(required(artifacts, "source", "artifacts")),
        "specification": resolve(required(artifacts, "specification", "artifacts")),
        **{f"testbench[{i}]": resolve(item) for i, item in enumerate(required(artifacts, "testbench", "artifacts"))},
        **{f"header[{i}]": resolve(item) for i, item in enumerate(artifacts.get("headers", []))},
        **{f"build_file[{i}]": resolve(item) for i, item in enumerate(artifacts.get("build_files", []))},
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    required_budgets = ("max_iterations", "max_csim_calls", "max_cosim_calls", "max_synthesis_calls", "max_model_calls")
    invalid_budgets = []
    for key in required_budgets:
        value = required(budgets, key, "budgets")
        if not isinstance(value, int) or value < 0:
            invalid_budgets.append(f"{key}={value!r}")
    checks = {
        "task_artifacts_exist": not missing, "top_function_declared": bool(interface.get("top_function")),
        "tool_declared": bool(target.get("tool")), "tool_version_declared": bool(target.get("tool_version")),
        "part_declared": bool(target.get("part")),
        "clock_period_valid": isinstance(target.get("clock_period_ns"), (int, float)) and target["clock_period_ns"] > 0,
        "budgets_valid": not invalid_budgets,
        "model_declared": bool(task["model"].get("provider")) and bool(task["model"].get("name")),
    }
    report = {"task_id": task["task_id"], "task_kind": task["task_kind"],
              "manifest": relative(task_path) if task_path.is_relative_to(REPO_ROOT) else str(task_path),
              "checks": checks, "missing_files": missing, "invalid_budgets": invalid_budgets,
              "passed": all(checks.values())}
    write_json(resolve(task["output_dir"]) / "task_validation.json", report)
    return report


def run_python(arguments: list[str], title: str) -> int:
    command = [sys.executable, *[str(resolve(item)) if i == 0 else item for i, item in enumerate(arguments)]]
    print(f"\n=== {title} ===\nCommand: {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def model_usage(adapter_output: Path) -> dict[str, Any]:
    records, totals = [], {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for path in sorted(adapter_output.glob("candidate_*_model_metadata.json")):
        data = load_json(path)
        records.append({"file": relative(path), **data})
        totals["model_calls"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if isinstance(data.get(key), int): totals[key] += data[key]
    return {"totals": totals, "calls": records}


def derive_phase(summary: dict[str, Any], task: dict[str, Any]) -> str:
    candidates = summary.get("candidates", [])
    if not candidates: return "diagnose_initial_design"
    verdict = candidates[-1].get("verdict")
    if verdict in {"reject_static", "reject_csim", "reject_duplicate"}: return "repair_correctness"
    if summary.get("budget", {}).get("synthesis_calls_remaining", 0) <= 0: return "terminated_budget"
    if len(candidates) >= task["budgets"]["max_iterations"]: return "terminated_iteration_limit"
    if verdict in {"keep_pareto_candidate", "accept_dominates_baseline", "reject_no_change", "reject_no_performance_gain"}: return "optimise_ppa"
    return "continue_evaluation"


def write_ledger(task_path: Path, task: dict[str, Any], summary: dict[str, Any] | None, status: str) -> Path:
    output_dir, adapter = resolve(task["output_dir"]), task["adapter"]
    adapter_output = resolve(load_json(resolve(adapter["config"]))["output_dir"])
    usage = model_usage(adapter_output)
    tool_usage = {"csim_calls": sum(1 for path in adapter_output.glob("candidate_*_csim_validation.json") if load_json(path).get("passed") is not None),
                  "cosim_calls": sum(1 for _ in adapter_output.glob("candidate_*_cosim_validation.json")),
                  "synthesis_calls": (summary or {}).get("budget", {}).get("synthesis_calls_used", 0)}
    budgets = task["budgets"]
    ledger = {"task_id": task["task_id"], "manifest": relative(task_path) if task_path.is_relative_to(REPO_ROOT) else str(task_path),
              "status": status, "updated_at": datetime.now(timezone.utc).isoformat(), "initial_condition": task["task_kind"],
              "policy": {"correctness_before_ppa": True, "duplicate_rejection_before_tools": True, "pareto_archive": True, "budgeted_termination": True},
              "budgets": budgets, "usage": {**tool_usage, **usage["totals"]},
              "remaining": {"csim_calls": max(0, budgets["max_csim_calls"] - tool_usage["csim_calls"]), "cosim_calls": max(0, budgets["max_cosim_calls"] - tool_usage["cosim_calls"]), "synthesis_calls": max(0, budgets["max_synthesis_calls"] - tool_usage["synthesis_calls"]), "model_calls": max(0, budgets["max_model_calls"] - usage["totals"]["model_calls"])},
              "experiment_summary": summary, "model_call_records": usage["calls"]}
    path = output_dir / "run_ledger.json"
    write_json(path, ledger)
    return path


def run_task(task_path: Path, *, status_only: bool = False, max_agent_steps: int | None = None) -> Path:
    """Execute the autonomous controller and return its ledger path."""
    task_path, task = task_path.resolve(), load_json(task_path.resolve())
    if task.get("adapter", {}).get("kind") != "legacy_ppa": raise ValueError("This controller currently requires a legacy_ppa adapter.")
    report = validate_task(task_path)
    _print_validation(report, resolve(task["output_dir"]) / "task_validation.json")
    if not report["passed"]: raise SystemExit(1)
    adapter, summary_path = task["adapter"], resolve(task["adapter"]["summary"])
    if not summary_path.is_file() and not status_only:
        if run_python(adapter["initialise_command"], "Diagnose and initialise baseline") != 0:
            write_ledger(task_path, task, None, "failed_initialisation"); raise SystemExit(1)
    summary = load_json(summary_path) if summary_path.is_file() else None
    if status_only: return write_ledger(task_path, task, summary, "status_only")
    previous_fingerprint, final_status = None, "terminated_no_progress"
    for step in range(1, (max_agent_steps or task["budgets"]["max_iterations"]) + 1):
        summary, phase = load_json(summary_path), derive_phase(load_json(summary_path), task)
        fingerprint = json.dumps({"phase": phase, "candidate_count": len(summary.get("candidates", [])), "budget": summary.get("budget", {}), "latest": summary.get("candidates", [])[-1] if summary.get("candidates") else None}, sort_keys=True)
        print(f"\nTrack A step {step}: {phase}")
        if phase.startswith("terminated_"): final_status = phase; break
        if fingerprint == previous_fingerprint: break
        previous_fingerprint = fingerprint
        if run_python(adapter["iteration_command"], "Autonomous repair/optimisation iteration") != 0: final_status = "iteration_failed"; break
        updated = load_json(summary_path)
        if json.dumps(updated, sort_keys=True) == json.dumps(summary, sort_keys=True): break
        final_status = derive_phase(updated, task)
    return write_ledger(task_path, task, load_json(summary_path), final_status)


def _print_validation(report: dict[str, Any], report_path: Path) -> None:
    print(f"\nTrack A task validation\nTask: {report['task_id']}\nInitial condition: {report['task_kind']}")
    for name, passed in report["checks"].items(): print(f"{'PASS' if passed else 'FAIL'}: {name}")
    for item in report["missing_files"]: print(f"Missing: {item}")
    for item in report["invalid_budgets"]: print(f"Invalid budget: {item}")
    print(f"Report: {relative(report_path)}\nOverall: {'PASS' if report['passed'] else 'FAIL'}")


def main_validate() -> None:
    parser = argparse.ArgumentParser(description="Validate a competition Track A task manifest."); parser.add_argument("task", type=Path)
    args = parser.parse_args(); report = validate_task(args.task); _print_validation(report, resolve(load_json(args.task.resolve())["output_dir"]) / "task_validation.json")
    if not report["passed"]: raise SystemExit(1)


def main_prepare() -> None:
    parser = argparse.ArgumentParser(description="Create a clean, isolated workspace for one Track A agent run."); parser.add_argument("task", type=Path); parser.add_argument("--run-id", required=True); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); run = prepare_fresh_run(args.task, args.run_id, force=args.force)
    print(f"\nFresh Track A workspace\nParent task: {run.task_id}\nRun ID: {run.run_id}\nManifest: {relative(run.manifest_path)}\nAdapter config: {relative(run.adapter_config_path)}\nCandidate workspace: {relative(run.candidate_dir)}\nLedger workspace: {relative(run.ledger_dir)}\nBaseline project: {relative(run.candidate_dir / 'baseline_project')}\nInherited candidate evidence: none\nInherited baseline synthesis evidence: none")


def main_run() -> None:
    parser = argparse.ArgumentParser(description="Run a budgeted FPT Track A LLM4HLS task."); parser.add_argument("task", type=Path); parser.add_argument("--status-only", action="store_true"); parser.add_argument("--max-agent-steps", type=int, default=None)
    args = parser.parse_args(); ledger = run_task(args.task, status_only=args.status_only, max_agent_steps=args.max_agent_steps)
    if args.status_only: print(f"\nLedger: {relative(ledger)}"); return
    print(f"\n=== Track A run complete ===\nLedger: {relative(ledger)}\nThe ledger records per-phase model tokens, tool calls, budgets, verdicts, and Pareto results.")


def main_fresh_run() -> None:
    parser = argparse.ArgumentParser(description="Materialise and execute a clean FPT Track A agent run."); parser.add_argument("task", type=Path); parser.add_argument("--run-id", required=True); parser.add_argument("--max-agent-steps", type=int, default=None); parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(); run = prepare_fresh_run(args.task, args.run_id)
    print(f"\nPrepared manifest: {relative(run.manifest_path)}")
    if args.prepare_only: return
    ledger = run_task(run.manifest_path, max_agent_steps=args.max_agent_steps)
    print(f"\n=== Fresh Track A execution finished ===\nManifest: {relative(run.manifest_path)}\nLedger: {relative(ledger)}")
