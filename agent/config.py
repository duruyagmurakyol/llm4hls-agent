#!/usr/bin/env python3

"""Load and validate unified LLM4HLS task manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent

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

SUPPORTED_ADAPTERS = {
    "auto",
    "autonomous_ppa",
    "direct_api_repair",
    "legacy_ppa",
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
    validate_task_paths(data)
    return TaskManifest(path=resolved, data=data)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_command(section: dict[str, Any], key: str, field: str) -> None:
    command = section.get(key)
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise ValueError(f"{field}.{key} must be a non-empty list of strings")


def _require_file_list(
    repair: dict[str, Any],
    key: str,
    *,
    required: bool,
) -> list[str]:
    files = repair.get(key)
    if files is None and not required:
        return []
    if not isinstance(files, list) or not files or not all(
        isinstance(item, str) and item for item in files
    ):
        raise ValueError(f"repair.{key} must be a non-empty list of strings")
    return files


def _validate_direct_api_repair(
    data: dict[str, Any],
    adapter: dict[str, Any],
) -> None:
    if "config" in adapter:
        raise ValueError(
            "direct_api_repair must be configured directly in the task manifest"
        )

    repair = data.get("repair")
    if not isinstance(repair, dict):
        raise ValueError("repair must be an object for repair-capable tasks")

    missing = sorted(REQUIRED_REPAIR_FIELDS - repair.keys())
    if missing:
        raise ValueError("Missing repair fields: " + ", ".join(missing))

    benchmark_source = repair["benchmark_source"]
    if not isinstance(benchmark_source, str) or not benchmark_source.strip():
        raise ValueError("repair.benchmark_source must be a non-empty string")

    editable = _require_file_list(repair, "editable_files", required=True)
    protected = _require_file_list(repair, "protected_files", required=True)
    _require_file_list(repair, "context_files", required=False)

    overlap = sorted(set(editable) & set(protected))
    if overlap:
        raise ValueError(
            "repair editable and protected files overlap: " + ", ".join(overlap)
        )

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


def _validate_autonomous_ppa(
    data: dict[str, Any],
    adapter: dict[str, Any],
) -> None:
    if "config" in adapter:
        raise ValueError(
            "autonomous_ppa must be configured directly in the task manifest"
        )

    optimisation = data.get("optimisation")
    if not isinstance(optimisation, dict):
        raise ValueError("optimisation must be an object for optimisation-capable tasks")

    prompt_constraints = optimisation.get("prompt_constraints", [])
    if not isinstance(prompt_constraints, list) or not all(
        isinstance(item, str) and item for item in prompt_constraints
    ):
        raise ValueError("optimisation.prompt_constraints must be a list of strings")

    validation = optimisation.get("validation", {})
    if not isinstance(validation, dict):
        raise ValueError("optimisation.validation must be an object")


def _requires_auto_cosim_budget(data: dict[str, Any]) -> bool:
    policy = data.get("validation_policy")
    if isinstance(policy, dict) and "requires_cosim" in policy:
        return bool(policy["requires_cosim"])

    track_a = data.get("track_a")
    if isinstance(track_a, dict) and "requires_cosim" in track_a:
        return bool(track_a["requires_cosim"])

    # Preserve the historical auto-manifest rule unless a task explicitly
    # declares that co-simulation is not part of its verification contract.
    return True


def _validate_auto(data: dict[str, Any], adapter: dict[str, Any]) -> None:
    if "config" in adapter:
        raise ValueError("auto tasks must be configured directly in the task manifest")

    _validate_direct_api_repair(data, adapter)
    _validate_autonomous_ppa(data, adapter)

    budgets = data["budgets"]
    for key in ("max_csim_calls", "max_synthesis_calls"):
        if budgets[key] < 2:
            raise ValueError(
                f"budgets.{key} must be at least 2 for auto detection and repair verification"
            )

    if _requires_auto_cosim_budget(data) and budgets["max_cosim_calls"] < 2:
        raise ValueError(
            "budgets.max_cosim_calls must be at least 2 for auto detection and repair verification"
        )


def _resolve_task_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def validate_task_paths(data: dict[str, Any]) -> None:
    artifacts = data["artifacts"]
    artifact_paths = [artifacts["source"], *artifacts["testbench"]]
    for value in artifact_paths:
        if not _resolve_task_path(value).is_file():
            raise ValueError(f"task file does not exist: {value}")

    adapter = data["adapter"]
    if adapter["kind"] not in {"direct_api_repair", "auto"}:
        return

    repair = data["repair"]
    benchmark_source = _resolve_task_path(str(repair["benchmark_source"]))
    if not benchmark_source.is_dir():
        raise ValueError(
            "repair.benchmark_source does not exist or is not a directory: "
            f"{repair['benchmark_source']}"
        )

    for key in ("editable_files", "protected_files", "context_files"):
        for relative_path in repair.get(key, []):
            if not (benchmark_source / relative_path).is_file():
                raise ValueError(f"repair file does not exist: {relative_path}")


def validate_task(data: dict[str, Any]) -> None:
    required = {
        "task_id",
        "task_kind",
        "artifacts",
        "interface",
        "target",
        "budgets",
        "model",
        "adapter",
        "output_dir",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError("Missing task fields: " + ", ".join(missing))

    artifacts = data["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be an object")

    source = artifacts.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("artifacts.source must be a non-empty string")

    testbench = artifacts.get("testbench")
    if not isinstance(testbench, list) or not testbench or not all(
        isinstance(item, str) and item.strip() for item in testbench
    ):
        raise ValueError("artifacts.testbench must be a non-empty list of strings")

    interface = data["interface"]
    top_function = interface.get("top_function") if isinstance(interface, dict) else None
    if not isinstance(top_function, str) or not top_function.strip():
        raise ValueError("interface.top_function must be a non-empty string")

    target = data["target"]
    if not isinstance(target, dict):
        raise ValueError("target must be an object")

    clock = target.get("clock_period_ns")
    if not _is_number(clock) or clock <= 0:
        raise ValueError("target.clock_period_ns must be a positive number")

    minimum_frequency = target.get("minimum_frequency_mhz")
    if not _is_number(minimum_frequency) or minimum_frequency <= 0:
        raise ValueError("target.minimum_frequency_mhz must be a positive number")

    platform = target.get("platform")
    part = target.get("part")
    has_platform = isinstance(platform, str) and bool(platform.strip())
    has_part = isinstance(part, str) and bool(part.strip())
    if not has_platform and not has_part:
        raise ValueError("target must define a non-empty platform or part")

    budgets = data["budgets"]
    if not isinstance(budgets, dict):
        raise ValueError("budgets must be an object")
    missing_budgets = sorted(REQUIRED_BUDGETS - budgets.keys())
    if missing_budgets:
        raise ValueError("Missing budget fields: " + ", ".join(missing_budgets))

    for key in REQUIRED_BUDGETS:
        value = budgets[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"budgets.{key} must be a non-negative integer")

    if budgets["max_iterations"] == 0:
        raise ValueError("budgets.max_iterations must be greater than zero")

    max_total_tokens = budgets.get("max_total_tokens")
    if max_total_tokens is not None and (
        isinstance(max_total_tokens, bool)
        or not isinstance(max_total_tokens, int)
        or max_total_tokens <= 0
    ):
        raise ValueError("budgets.max_total_tokens must be null or a positive integer")

    adapter = data["adapter"]
    if not isinstance(adapter, dict) or not isinstance(adapter.get("kind"), str):
        raise ValueError("adapter.kind is required")

    adapter_kind = adapter["kind"].strip()
    if adapter_kind not in SUPPORTED_ADAPTERS:
        raise ValueError(f"Unsupported adapter kind: {adapter_kind or '<empty>'}")

    if adapter_kind == "auto":
        _validate_auto(data, adapter)
    elif adapter_kind == "direct_api_repair":
        _validate_direct_api_repair(data, adapter)
    elif adapter_kind == "autonomous_ppa":
        _validate_autonomous_ppa(data, adapter)
    elif adapter_kind == "legacy_ppa":
        config = adapter.get("config")
        if not isinstance(config, str) or not config.strip():
            raise ValueError("legacy_ppa requires adapter.config")
