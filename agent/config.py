#!/usr/bin/env python3

"""Load and validate unified LLM4HLS task manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_BUDGETS = {
    "max_iterations",
    "max_csim_calls",
    "max_cosim_calls",
    "max_synthesis_calls",
    "max_model_calls",
}

REQUIRED_REPAIR_FIELDS = {
    "benchmark_source",
    "editable_files",
    "protected_files",
    "host_validation",
    "independent_validation",
}


@dataclass(frozen=True)
class TaskManifest:
    path: Path
    data: dict[str, Any]

    @property
    def task_id(self) -> str:
        return str(self.data["task_id"])

    @property
    def output_dir(self) -> Path:
        return Path(str(self.data["output_dir"]))

    @property
    def adapter_kind(self) -> str:
        return str(self.data["adapter"]["kind"])


def load_task(path: Path) -> TaskManifest:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Task manifest not found: {resolved}")

    data = json.loads(resolved.read_text(encoding="utf-8"))
    validate_task(data)
    return TaskManifest(path=resolved, data=data)


def _require_command(section: dict[str, Any], key: str, field: str) -> None:
    command = section.get(key)
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError(f"{field}.{key} must be a non-empty list of strings")


def _validate_direct_api_repair(data: dict[str, Any], adapter: dict[str, Any]) -> None:
    if "config" in adapter:
        raise ValueError("direct_api_repair must be configured directly in the task manifest")

    repair = data.get("repair")
    if not isinstance(repair, dict):
        raise ValueError("repair must be an object for direct_api_repair tasks")

    missing = sorted(REQUIRED_REPAIR_FIELDS - repair.keys())
    if missing:
        raise ValueError("Missing repair fields: " + ", ".join(missing))

    for key in ("editable_files", "protected_files"):
        files = repair[key]
        if not isinstance(files, list) or not files or not all(isinstance(item, str) and item for item in files):
            raise ValueError(f"repair.{key} must be a non-empty list of strings")

    host = repair["host_validation"]
    if not isinstance(host, dict):
        raise ValueError("repair.host_validation must be an object")
    _require_command(host, "command", "repair.host_validation")
    _require_command(host, "run_command", "repair.host_validation")

    independent = repair["independent_validation"]
    if not isinstance(independent, dict):
        raise ValueError("repair.independent_validation must be an object")
    if independent.get("enabled", False):
        _require_command(independent, "command", "repair.independent_validation")


def validate_task(data: dict[str, Any]) -> None:
    required = {"task_id", "task_kind", "artifacts", "interface", "target", "budgets", "model", "adapter", "output_dir"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError("Missing task fields: " + ", ".join(missing))

    artifacts = data["artifacts"]
    if not isinstance(artifacts, dict) or "source" not in artifacts or "testbench" not in artifacts:
        raise ValueError("artifacts must define source and testbench")

    interface = data["interface"]
    if not isinstance(interface, dict) or not interface.get("top_function"):
        raise ValueError("interface.top_function is required")

    budgets = data["budgets"]
    if not isinstance(budgets, dict):
        raise ValueError("budgets must be an object")
    missing_budgets = sorted(REQUIRED_BUDGETS - budgets.keys())
    if missing_budgets:
        raise ValueError("Missing budget fields: " + ", ".join(missing_budgets))

    for key in REQUIRED_BUDGETS:
        value = budgets[key]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"budgets.{key} must be a non-negative integer")

    adapter = data["adapter"]
    if not isinstance(adapter, dict) or not adapter.get("kind"):
        raise ValueError("adapter.kind is required")

    if adapter["kind"] == "direct_api_repair":
        _validate_direct_api_repair(data, adapter)
