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
