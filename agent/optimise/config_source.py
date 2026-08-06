"""In-memory compatibility layer for the existing PPA optimisation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from agent.config import TaskManifest


@dataclass(frozen=True)
class InMemoryConfig:
    """Path-like JSON source backed by task data instead of a file."""

    data: dict[str, Any]
    identity: str

    def resolve(self) -> "InMemoryConfig":
        return self

    def is_file(self) -> bool:
        return True

    def read_text(self, encoding: str = "utf-8") -> str:
        del encoding
        return json.dumps(self.data)

    def __str__(self) -> str:
        return f"in-memory:{self.identity}"


ConfigSource: TypeAlias = Path | InMemoryConfig
ConfigInput: TypeAlias = Path | TaskManifest | dict[str, Any]


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _saved_baseline_metrics(output_dir: Path) -> dict[str, Any]:
    """Recover existing original-baseline metrics without invoking Vitis.

    Track-A runs persist the untouched original kernel separately from the
    promoted/selected design. Prefer that scoring baseline, then the durable
    archive, and only then the verified baseline used by older runs.
    """

    original = _load_json_object(output_dir / "original_scoring_baseline.json")
    metrics = original.get("metrics")
    if isinstance(metrics, dict) and metrics:
        return dict(metrics)

    state = _load_json_object(output_dir / "candidate_state.json")
    archived = state.get("original_baseline")
    if isinstance(archived, dict):
        metrics = archived.get("metrics")
        if isinstance(metrics, dict) and metrics:
            return dict(metrics)

    verified = _load_json_object(output_dir / "verified_baseline.json")
    metrics = verified.get("metrics")
    if isinstance(metrics, dict) and metrics:
        return dict(metrics)

    return {}


def ppa_config_from_task(task: TaskManifest) -> dict[str, Any]:
    """Translate one authoritative task manifest into the existing PPA shape."""

    artifacts = task.data["artifacts"]
    build_files = artifacts.get("build_files") or []
    if len(build_files) != 1:
        raise ValueError("PPA tasks must define exactly one build file")

    optimisation = task.data.get("optimisation") or {}
    validation = {
        "constant_loop_tail_bounds": True,
        "preserve_diagnosed_loop_label": True,
        **optimisation.get("validation", {}),
    }

    protected_contract = task.data["interface"].get("protected_contract", [])
    configured_constraints = optimisation.get("prompt_constraints", [])
    prompt_constraints = [*protected_contract, *configured_constraints]
    if not prompt_constraints:
        prompt_constraints = [
            "Preserve the top-level function signature and all testbench-observed semantics.",
            "Do not modify the supplied testbench or baseline source in place.",
        ]

    task_root = task.data.get("task_root")
    benchmark = (
        Path(str(task_root)).name
        if task_root
        else Path(str(artifacts["source"])).parent.parent.name
    )
    target = task.data.get("target") or {}
    target_clock_period_ns = float(target.get("clock_period_ns", 10.0))
    minimum_frequency_mhz = float(target.get("minimum_frequency_mhz", 100.0))
    budgets = task.data["budgets"]
    max_candidates = int(budgets["max_iterations"])
    track_a = task.data.get("track_a")
    requires_cosim = (
        bool(track_a.get("requires_cosim", False))
        if isinstance(track_a, dict)
        else int(budgets.get("max_cosim_calls", 0)) > 0
    )
    selection = dict(optimisation.get("selection") or {})
    if isinstance(track_a, dict):
        # Track-A still records the reference-harness score separately, but
        # final design choice defaults to the richer multi-objective Pareto
        # policy unless a manifest explicitly requests another mode.
        selection.setdefault("mode", "research_pareto")

    output_dir = Path(str(task.output_dir)).expanduser()
    baseline: dict[str, Any] = {
        "source": artifacts["source"],
        "tcl": build_files[0],
        "project_dir": optimisation.get(
            "baseline_project_dir",
            f"/tmp/llm4hls-agent/{task.task_id}_baseline",
        ),
    }
    saved_metrics = _saved_baseline_metrics(output_dir)
    if saved_metrics:
        baseline["metrics"] = saved_metrics

    config: dict[str, Any] = {
        "experiment_name": f"{task.task_id}_ppa",
        "benchmark": benchmark,
        "top_function": task.data["interface"]["top_function"],
        "target_clock_period_ns": target_clock_period_ns,
        "minimum_frequency_mhz": minimum_frequency_mhz,
        "resource_limits": dict(target.get("resource_limits") or {}),
        "selection": selection,
        "requires_cosim": requires_cosim,
        "baseline": baseline,
        "validation": validation,
        "prompt_constraints": prompt_constraints,
        "output_dir": str(task.output_dir),
        "model": task.data["model"],
        "budget": {
            "max_candidates": max_candidates,
            "max_synthesis_calls": int(budgets["max_synthesis_calls"]),
            "max_cosim_calls": int(budgets.get("max_cosim_calls", max_candidates)),
        },
    }
    if isinstance(track_a, dict):
        config["track_a"] = dict(track_a)

    target_loop_label = optimisation.get("target_loop_label")
    if target_loop_label:
        config["target_loop_label"] = target_loop_label

    timeouts = optimisation.get("timeouts")
    if timeouts:
        config["timeouts"] = timeouts

    return config


def as_config_source(value: ConfigInput) -> ConfigSource:
    if isinstance(value, TaskManifest):
        return InMemoryConfig(ppa_config_from_task(value), value.task_id)
    if isinstance(value, dict):
        identity = str(value.get("experiment_name", "ppa"))
        return InMemoryConfig(value, identity)
    return value.expanduser().resolve()
