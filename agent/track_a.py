"""Compatibility frontend for the official FPT 2026 Track-A task package.

The competition package is treated as untrusted read-only input. Only the
public kernel, declared headers, public testbench, description and task.toml
are copied into a controlled staging directory. Hidden tests and reference
solutions are deliberately excluded from the agent-visible workspace.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import tomllib
from pathlib import Path
from typing import Any

from agent.config import TaskManifest
from agent.onboarding import REPO_ROOT


DEFAULT_PART = "xcu55c-fsvh2892-2L-e"
DEFAULT_CLOCK_NS = 5.0
DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B"
_SUPPORTED_TASK_TYPES = {
    "generate",
    "repair",
    "optimize",
    "synth_fix",
    "structural",
}


def is_track_a_task(root: Path) -> bool:
    """Return whether *root* has the official Track-A task-package marker."""

    return root.is_dir() and (root / "task.toml").is_file()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _safe_relative(root: Path, value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{field} must be relative to the task directory")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes the task directory: {value}") from error
    if not resolved.is_file():
        raise ValueError(f"{field} does not exist: {value}")
    return resolved


def _string_list(spec: dict[str, Any], key: str) -> list[str]:
    values = spec.get(key, [])
    if not isinstance(values, list) or not all(
        isinstance(item, str) and item.strip() for item in values
    ):
        raise ValueError(f"task.toml {key} must be a list of non-empty strings")
    return [item.strip() for item in values]


def _task_id(value: object, fallback: str) -> str:
    raw = str(value or fallback).strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not safe:
        raise ValueError("task.toml task_id is empty after path sanitisation")
    return safe


def _positive_int(value: object, *, field: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"task.toml {field} must be a positive integer")
    return value


def _positive_float(value: object, *, field: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"task.toml {field} must be a positive number")
    return float(value)


def _model_config() -> dict[str, Any]:
    max_total = os.environ.get("LLM4HLS_MAX_TOTAL_TOKENS")
    return {
        "provider": os.environ.get("LLM4HLS_PROVIDER", "siliconflow"),
        "name": os.environ.get("LLM4HLS_MODEL", DEFAULT_MODEL),
        "temperature": float(os.environ.get("LLM4HLS_TEMPERATURE", "0")),
        "max_tokens": int(os.environ.get("LLM4HLS_MAX_OUTPUT_TOKENS", "4096")),
        "timeout_seconds": int(os.environ.get("LLM4HLS_API_TIMEOUT_SECONDS", "180")),
        "enable_thinking": False,
        "max_total_tokens": int(max_total) if max_total else None,
    }


def _budgets(credit_budget: int) -> dict[str, Any]:
    """Translate the package hint into bounded controller call limits.

    The reference harness's weighted credits are not the competition's only
    possible evaluator contract, so they are preserved as metadata rather than
    pretending they are identical to this agent's call counters.
    """

    iterations = max(2, min(6, credit_budget // 8 or 2))
    max_total = os.environ.get("LLM4HLS_MAX_TOTAL_TOKENS")
    return {
        "max_iterations": iterations,
        "max_csim_calls": max(4, iterations + 2),
        "max_cosim_calls": max(2, min(4, iterations)),
        "max_synthesis_calls": max(3, iterations + 1),
        "max_model_calls": iterations,
        "max_total_tokens": int(max_total) if max_total else None,
    }


def _write_cfg(
    path: Path,
    *,
    kernel: str,
    public_tb: str,
    top: str,
    part: str,
    clock_ns: float,
) -> None:
    path.write_text(
        "\n".join(
            [
                "[hls]",
                "flow_target=vivado",
                f"syn.file={kernel}",
                "syn.cflags=-I.",
                f"syn.top={top}",
                f"tb.file={public_tb}",
                "tb.cflags=-I.",
                f"part={part}",
                f"clock={clock_ns:g}ns",
                "",
            ]
        ),
        encoding="utf-8",
    )


def resolve_track_a_task(root: Path) -> TaskManifest:
    """Stage a public Track-A package and return the existing auto manifest."""

    root = root.expanduser().resolve()
    if not is_track_a_task(root):
        raise ValueError(f"Track-A task.toml not found in {root}")

    spec = tomllib.loads((root / "task.toml").read_text(encoding="utf-8"))
    task_id = _task_id(spec.get("task_id"), root.name)
    task_type = str(spec.get("task_type", "generate")).strip()
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError(
            "task.toml task_type must be one of: "
            + ", ".join(sorted(_SUPPORTED_TASK_TYPES))
        )

    top = str(spec.get("top", "")).strip()
    if not top:
        raise ValueError("task.toml top must be a non-empty string")

    kernel_name = str(spec.get("kernel_file", "")).strip()
    public_tb_name = str(spec.get("public_tb", "")).strip()
    if not kernel_name or not public_tb_name:
        raise ValueError("task.toml must define kernel_file and public_tb")

    header_names = _string_list(spec, "header_files")
    public_names = [kernel_name, *header_names, public_tb_name]
    public_sources = {
        name: _safe_relative(root, name, f"task.toml file {name}")
        for name in public_names
    }

    description_source = root / "description.md"
    if description_source.exists() and not description_source.is_file():
        raise ValueError("description.md exists but is not a regular file")

    target = spec.get("target", {})
    if not isinstance(target, dict):
        raise ValueError("task.toml [target] must be a table")
    part = str(target.get("part") or DEFAULT_PART).strip()
    clock_ns = _positive_float(
        target.get("clock_ns"), field="target.clock_ns", default=DEFAULT_CLOCK_NS
    )
    difficulty = _positive_int(spec.get("difficulty"), field="difficulty", default=1)
    credit_budget = _positive_int(spec.get("budget"), field="budget", default=40)
    requires_cosim = bool(spec.get("requires_cosim", False))

    staging_root = REPO_ROOT / "experiments" / "track_a_staging" / task_id
    output_dir = REPO_ROOT / "experiments" / "track_a" / task_id
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    for relative, source in public_sources.items():
        destination = staging_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    shutil.copy2(root / "task.toml", staging_root / "task.toml")
    if description_source.is_file():
        shutil.copy2(description_source, staging_root / "description.md")
    else:
        (staging_root / "description.md").write_text("", encoding="utf-8")

    cfg_path = staging_root / "task.cfg"
    _write_cfg(
        cfg_path,
        kernel=kernel_name,
        public_tb=public_tb_name,
        top=top,
        part=part,
        clock_ns=clock_ns,
    )

    protected = [
        *header_names,
        public_tb_name,
        "description.md",
        "task.toml",
        "task.cfg",
    ]
    context = [*header_names, public_tb_name, "description.md"]
    host_command = [
        "g++",
        "-std=c++17",
        "-I",
        ".",
        kernel_name,
        public_tb_name,
        "-o",
        ".agent_host_test",
    ]
    independent_command = [
        "bash",
        "-lc",
        "cd '{workspace}' && "
        + "vitis-run --mode hls --csim --config "
        + shlex.quote("task.cfg")
        + " --work_dir vitis_work",
    ]
    model = _model_config()
    budgets = _budgets(credit_budget)

    data: dict[str, Any] = {
        "task_id": f"track_a_{task_id}",
        "task_kind": task_type,
        "task_root": _portable_path(staging_root),
        "artifacts": {
            "source": _portable_path(staging_root / kernel_name),
            "testbench": [_portable_path(staging_root / public_tb_name)],
            "headers": [
                _portable_path(staging_root / name) for name in header_names
            ],
            "build_files": [_portable_path(cfg_path)],
        },
        "interface": {
            "top_function": top,
            "language": "cpp",
            "numerical_tolerance": None,
            "protected_contract": [
                "Preserve the declared top-level function signature.",
                "Modify only the declared kernel source file.",
                "Treat description.md, headers, testbench and task.toml as read-only.",
            ],
        },
        "target": {
            "tool": "AMD Vitis HLS",
            "tool_version": "2025.2",
            "part": part,
            "clock_period_ns": clock_ns,
            "minimum_frequency_mhz": 100.0,
            "resource_limits": {},
        },
        "budgets": budgets,
        "model": model,
        "repair": {
            "benchmark_source": _portable_path(staging_root),
            "editable_files": [kernel_name],
            "protected_files": protected,
            "context_files": context,
            "repair_constraints": [
                "Satisfy the public specification in description.md.",
                "Do not use hidden tests or reference solutions.",
                "Preserve the top-level interface and public-test behaviour.",
            ],
            "host_validation": {
                "command": host_command,
                "run_command": ["./.agent_host_test"],
            },
            "independent_validation": {
                "enabled": True,
                "command": independent_command,
            },
        },
        "optimisation": {
            "prompt_constraints": [
                "Satisfy description.md and preserve public-test behaviour.",
                "Modify only the kernel source; never modify fixed package files.",
                "Optimise only after correctness is established.",
            ],
            "validation": {},
        },
        "adapter": {"kind": "auto"},
        "output_dir": _portable_path(output_dir),
        "track_a": {
            "source_package": str(root),
            "task_type": task_type,
            "difficulty": difficulty,
            "credit_budget": credit_budget,
            "requires_cosim": requires_cosim,
            "initial_condition": str(spec.get("initial_condition", "")),
            "hidden_and_reference_excluded": True,
        },
    }
    return TaskManifest(path=root / "task.toml", data=data)


def onboard_track_a_task(root: Path) -> TaskManifest:
    """Resolve and print a concise competition-package onboarding report."""

    task = resolve_track_a_task(root)
    meta = task.data["track_a"]
    print("Track-A task-package discovery")
    print(f"Task: {task.task_id}")
    print(f"Task type: {meta['task_type']}")
    print(f"Top function: {task.data['interface']['top_function']}")
    print(f"Source: {task.data['artifacts']['source']}")
    print(f"Public testbench: {task.data['artifacts']['testbench'][0]}")
    print(f"Part: {task.data['target']['part']}")
    print(f"Clock: {task.data['target']['clock_period_ns']:g} ns")
    print(f"Reference credit hint: {meta['credit_budget']}")
    print("Hidden tests/reference solutions copied: no")
    print(f"Staged public package: {task.data['task_root']}")
    return task
