#!/usr/bin/env python3

"""Validate an FPT 2026 Track A task package before an autonomous run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing {context}.{key}")
    return mapping[key]


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a competition Track A task manifest.")
    parser.add_argument("task", type=Path)
    args = parser.parse_args()

    task_path = args.task.resolve()
    task = load_json(task_path)

    for key in ("task_id", "task_kind", "artifacts", "interface", "target", "budgets", "model", "output_dir"):
        require(task, key, "task")

    artifacts = task["artifacts"]
    interface = task["interface"]
    target = task["target"]
    budgets = task["budgets"]

    source = resolve(require(artifacts, "source", "artifacts"))
    specification = resolve(require(artifacts, "specification", "artifacts"))
    testbenches = [resolve(item) for item in require(artifacts, "testbench", "artifacts")]
    headers = [resolve(item) for item in artifacts.get("headers", [])]
    build_files = [resolve(item) for item in artifacts.get("build_files", [])]

    paths = {
        "source": source,
        "specification": specification,
        **{f"testbench[{i}]": path for i, path in enumerate(testbenches)},
        **{f"header[{i}]": path for i, path in enumerate(headers)},
        **{f"build_file[{i}]": path for i, path in enumerate(build_files)},
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]

    required_budget_keys = (
        "max_iterations",
        "max_csim_calls",
        "max_cosim_calls",
        "max_synthesis_calls",
        "max_model_calls",
    )
    invalid_budgets = []
    for key in required_budget_keys:
        value = require(budgets, key, "budgets")
        if not isinstance(value, int) or value < 0:
            invalid_budgets.append(f"{key}={value!r}")

    checks = {
        "task_artifacts_exist": not missing,
        "top_function_declared": bool(interface.get("top_function")),
        "tool_declared": bool(target.get("tool")),
        "tool_version_declared": bool(target.get("tool_version")),
        "part_declared": bool(target.get("part")),
        "clock_period_valid": isinstance(target.get("clock_period_ns"), (int, float)) and target["clock_period_ns"] > 0,
        "budgets_valid": not invalid_budgets,
        "model_declared": bool(task["model"].get("provider")) and bool(task["model"].get("name")),
    }

    output_dir = resolve(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "task_validation.json"
    report = {
        "task_id": task["task_id"],
        "task_kind": task["task_kind"],
        "manifest": str(task_path.relative_to(REPO_ROOT)) if task_path.is_relative_to(REPO_ROOT) else str(task_path),
        "checks": checks,
        "missing_files": missing,
        "invalid_budgets": invalid_budgets,
        "passed": all(checks.values()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nTrack A task validation")
    print(f"Task: {task['task_id']}")
    print(f"Initial condition: {task['task_kind']}")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    for item in missing:
        print(f"Missing: {item}")
    for item in invalid_budgets:
        print(f"Invalid budget: {item}")
    print(f"Report: {report_path.relative_to(REPO_ROOT)}")
    print(f"Overall: {'PASS' if report['passed'] else 'FAIL'}")

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
