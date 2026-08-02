"""Provenance-safe zero-configuration benchmark onboarding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent.onboarding import REPO_ROOT, DiscoveredBenchmark, discover_benchmark


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError as error:
        raise ValueError(f"Benchmark file must be inside the repository: {path}") from error


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_source(benchmark: DiscoveredBenchmark) -> tuple[Path, str]:
    source = benchmark.source.resolve()
    try:
        source.relative_to(benchmark.root)
        return source, "tcl_source_inside_benchmark"
    except ValueError:
        pass

    digest = _digest(source)
    equivalents = sorted(
        path.resolve()
        for path in benchmark.root.rglob(f"*{source.suffix}")
        if path.is_file() and _digest(path) == digest
    )
    if len(equivalents) == 1:
        return equivalents[0], "equivalent_source_inside_benchmark"
    if len(equivalents) > 1:
        rendered = ", ".join(_repo_relative(path) for path in equivalents)
        raise ValueError(
            "The TCL source is outside the benchmark and has multiple identical copies inside it: "
            + rendered
        )
    return source, "external_tcl_source_no_internal_equivalent"


def onboard_benchmark(root: Path) -> Path:
    benchmark = discover_benchmark(root)
    source, source_resolution = _canonical_source(benchmark)

    generated_dir = REPO_ROOT / "experiments" / "onboarding" / benchmark.name
    generated_dir.mkdir(parents=True, exist_ok=True)
    optimisation_path = generated_dir / "optimisation.json"
    task_path = generated_dir / "task.json"
    report_path = generated_dir / "onboarding_report.json"

    optimisation_output = f"experiments/onboarding/{benchmark.name}/autonomous_ppa"
    task_output = f"experiments/onboarding/{benchmark.name}/agent_result"

    model = {
        "provider": "siliconflow",
        "name": "Qwen/Qwen3.5-122B-A10B",
        "temperature": 0.0,
        "max_tokens": 4096,
        "enable_thinking": False,
    }
    optimisation: dict[str, Any] = {
        "experiment_name": f"auto_{benchmark.name}_ppa",
        "benchmark": benchmark.name,
        "top_function": benchmark.top_function,
        "baseline": {
            "source": _repo_relative(source),
            "tcl": _repo_relative(benchmark.tcl),
            "project_dir": _repo_relative(benchmark.project_dir),
        },
        "validation": {
            "constant_loop_tail_bounds": True,
            "preserve_diagnosed_loop_label": True,
        },
        "prompt_constraints": [
            "Preserve the top-level function signature and all testbench-observed semantics.",
            "Do not modify the supplied testbench or baseline source in place.",
        ],
        "output_dir": optimisation_output,
        "model": model,
        "budget": {"max_candidates": 5, "max_synthesis_calls": 4},
    }
    optimisation_path.write_text(json.dumps(optimisation, indent=2) + "\n", encoding="utf-8")

    task = {
        "task_id": f"auto_{benchmark.name}_001",
        "task_kind": "correct_unoptimised",
        "artifacts": {
            "source": _repo_relative(source),
            "testbench": [_repo_relative(path) for path in benchmark.testbenches],
            "headers": [_repo_relative(path) for path in benchmark.headers],
            "build_files": [_repo_relative(benchmark.tcl)],
        },
        "interface": {
            "top_function": benchmark.top_function,
            "language": "cpp",
            "numerical_tolerance": None,
            "protected_contract": [
                "Preserve the top-level function signature.",
                "Preserve output semantics checked by the supplied testbench.",
            ],
        },
        "target": {
            "tool": "AMD Vitis HLS",
            "tool_version": "2025.2",
            "part": benchmark.part,
            "clock_period_ns": benchmark.clock_period_ns,
            "resource_limits": {},
        },
        "budgets": {
            "max_iterations": 5,
            "max_csim_calls": 5,
            "max_cosim_calls": 0,
            "max_synthesis_calls": 4,
            "max_model_calls": 5,
            "max_total_tokens": None,
        },
        "model": model,
        "adapter": {
            "kind": "autonomous_ppa",
            "config": _repo_relative(optimisation_path),
        },
        "output_dir": task_output,
    }
    task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")

    report = {
        "benchmark": benchmark.name,
        "benchmark_root": _repo_relative(benchmark.root),
        "selected_tcl": _repo_relative(benchmark.tcl),
        "tcl_source": _repo_relative(benchmark.source),
        "canonical_source": _repo_relative(source),
        "source_resolution": source_resolution,
        "source_sha256": _digest(source),
        "top_function": benchmark.top_function,
        "part": benchmark.part,
        "clock_period_ns": benchmark.clock_period_ns,
        "testbenches": [_repo_relative(path) for path in benchmark.testbenches],
        "headers": [_repo_relative(path) for path in benchmark.headers],
        "optimisation_output_dir": optimisation_output,
        "task_output_dir": task_output,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Automatic benchmark onboarding")
    print(f"Benchmark: {benchmark.name}")
    print(f"TCL: {_repo_relative(benchmark.tcl)}")
    print(f"Top function: {benchmark.top_function}")
    print(f"Source: {_repo_relative(source)} ({source_resolution})")
    print(f"Testbench files: {len(benchmark.testbenches)}")
    print(f"Part: {benchmark.part}")
    print(f"Clock: {benchmark.clock_period_ns:g} ns")
    print(f"Generated task: {_repo_relative(task_path)}")
    print(f"Generated optimisation config: {_repo_relative(optimisation_path)}")
    print(f"Onboarding report: {_repo_relative(report_path)}")
    return task_path
