#!/usr/bin/env python3

"""Materialise an isolated Track A run from a reusable task manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def safe_run_id(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    if not value:
        raise ValueError("Run ID must contain at least one usable character")
    return value


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clean, isolated workspace for one Track A agent run."
    )
    parser.add_argument("task", type=Path, help="Reusable Track A task manifest")
    parser.add_argument("--run-id", required=True, help="Unique identifier for this run")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace generated configuration files only; candidate outputs are never deleted",
    )
    args = parser.parse_args()

    source_task_path = args.task.resolve()
    source_task = load_json(source_task_path)
    adapter = source_task.get("adapter") or {}
    if adapter.get("kind") != "legacy_ppa":
        raise ValueError("Fresh workspace preparation currently supports legacy_ppa adapters")

    run_id = safe_run_id(args.run_id)
    task_id = str(source_task["task_id"])
    generated_root = REPO_ROOT / "experiments" / "track_a_runs" / task_id / run_id
    generated_config_dir = generated_root / "config"
    generated_candidate_dir = generated_root / "candidates"
    generated_ledger_dir = generated_root / "ledger"
    generated_manifest_path = generated_config_dir / "task.json"
    generated_adapter_path = generated_config_dir / "ppa.json"

    if generated_manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"Run already exists: {generated_manifest_path}\n"
            "Choose a new --run-id. Existing run evidence is never overwritten."
        )

    adapter_config_path = Path(adapter["config"])
    if not adapter_config_path.is_absolute():
        adapter_config_path = REPO_ROOT / adapter_config_path
    adapter_config = load_json(adapter_config_path)

    if generated_candidate_dir.exists() and any(generated_candidate_dir.iterdir()):
        raise FileExistsError(
            f"Candidate workspace is not empty: {generated_candidate_dir}\n"
            "Choose a new --run-id; fresh runs never inherit or delete candidate evidence."
        )

    generated_candidate_dir.mkdir(parents=True, exist_ok=True)
    generated_ledger_dir.mkdir(parents=True, exist_ok=True)

    fresh_adapter_config = json.loads(json.dumps(adapter_config))
    fresh_adapter_config["experiment_name"] = f"{adapter_config['experiment_name']}__{run_id}"
    fresh_adapter_config["output_dir"] = relative(generated_candidate_dir)

    budgets = source_task["budgets"]
    fresh_adapter_config.setdefault("budget", {})
    fresh_adapter_config["budget"]["max_candidates"] = int(budgets["max_iterations"])
    fresh_adapter_config["budget"]["max_synthesis_calls"] = int(
        budgets["max_synthesis_calls"]
    )
    fresh_adapter_config["model"] = json.loads(json.dumps(source_task["model"]))
    write_json(generated_adapter_path, fresh_adapter_config)

    fresh_task = json.loads(json.dumps(source_task))
    fresh_task["task_id"] = f"{task_id}__{run_id}"
    fresh_task["parent_task_id"] = task_id
    fresh_task["run_id"] = run_id
    fresh_task["output_dir"] = relative(generated_ledger_dir)
    fresh_task["adapter"] = {
        "kind": "legacy_ppa",
        "config": relative(generated_adapter_path),
        "initialise_command": [
            "scripts/run_ppa_optimisation.py",
            relative(generated_adapter_path),
        ],
        "iteration_command": [
            "scripts/run_ppa_agent_iteration.py",
            relative(generated_adapter_path),
            "--allow-synthesis",
        ],
        "summary": relative(generated_candidate_dir / "experiment_summary.json"),
    }
    fresh_task["provenance"] = {
        "source_manifest": relative(source_task_path),
        "workspace_isolation": {
            "inherits_candidate_evidence": False,
            "inherits_model_responses": False,
            "inherits_validation_reports": False,
            "inherits_pareto_archive": False,
            "may_reuse_immutable_baseline_reports": True,
        },
    }
    write_json(generated_manifest_path, fresh_task)

    print("\nFresh Track A workspace")
    print(f"Parent task: {task_id}")
    print(f"Run ID: {run_id}")
    print(f"Manifest: {relative(generated_manifest_path)}")
    print(f"Adapter config: {relative(generated_adapter_path)}")
    print(f"Candidate workspace: {relative(generated_candidate_dir)}")
    print(f"Ledger workspace: {relative(generated_ledger_dir)}")
    print("Inherited candidate evidence: none")


if __name__ == "__main__":
    main()
