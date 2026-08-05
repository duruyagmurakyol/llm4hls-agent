#!/usr/bin/env python3

"""Run repeated repair-only or repair-to-optimisation agent tasks.

This is the generic entry point for suite indexes that contain either
``direct_api_repair`` tasks or unified ``auto`` tasks. It reuses the established
sequential runner and its partial-summary behaviour while extending preflight
validation to the full-agent adapter.
"""

from __future__ import annotations

from pathlib import Path

from scripts import run_overnight_repair_suite as suite

_original_validate_task_inputs = suite.validate_task_inputs


def validate_task_inputs(task_path: Path) -> list[str]:
    task = suite.load_json(task_path)
    adapter = task.get("adapter")
    kind = adapter.get("kind") if isinstance(adapter, dict) else None

    errors = _original_validate_task_inputs(task_path)
    adapter_error = "adapter.kind must be direct_api_repair"

    if kind == "auto":
        errors = [error for error in errors if error != adapter_error]
        if not isinstance(task.get("optimisation"), dict):
            errors.append("auto tasks must define task.optimisation")
    elif kind != "direct_api_repair":
        errors = [error for error in errors if error != adapter_error]
        errors.append("adapter.kind must be direct_api_repair or auto")

    return errors


suite.validate_task_inputs = validate_task_inputs


if __name__ == "__main__":
    suite.main()
