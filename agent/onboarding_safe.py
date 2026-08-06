"""Provenance-safe zero-configuration benchmark onboarding."""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

from agent.config import TaskManifest
from agent.onboarding import REPO_ROOT, DiscoveredBenchmark, discover_benchmark


DEFAULT_MODEL = {
    "provider": "siliconflow",
    "name": "Qwen/Qwen3.5-122B-A10B",
    "temperature": 0.0,
    "max_tokens": 4096,
    "timeout_seconds": 120,
    "enable_thinking": False,
}

DEFAULT_BUDGETS = {
    "max_iterations": 5,
    "max_csim_calls": 5,
    "max_cosim_calls": 0,
    "max_synthesis_calls": 4,
    "max_model_calls": 5,
    "max_total_tokens": None,
}

DEFAULT_OPTIMISATION = {
    "prompt_constraints": [],
    "validation": {},
    "selection": {"mode": "research_pareto"},
}


def _portable_path(path: Path) -> str:
    """Use repository-relative paths locally and absolute paths for mounted tasks."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _root_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise ValueError(f"Benchmark file must be inside the task directory: {path}") from error


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_source(benchmark: DiscoveredBenchmark) -> Path:
    source = benchmark.source.resolve()
    try:
        source.relative_to(benchmark.root)
        return source
    except ValueError:
        pass

    digest = _digest(source)
    equivalents = sorted(
        path.resolve()
        for path in benchmark.root.rglob(f"*{source.suffix}")
        if path.is_file() and _digest(path) == digest
    )
    if len(equivalents) == 1:
        return equivalents[0]
    if len(equivalents) > 1:
        rendered = ", ".join(_portable_path(path) for path in equivalents)
        raise ValueError(
            "The configured source is outside the benchmark and has multiple identical copies inside it: "
            + rendered
        )
    raise ValueError(
        "The configured source is outside the benchmark and no equivalent editable source exists inside it"
    )


def _independent_validation_command(build_file: str) -> list[str]:
    quoted = shlex.quote(build_file)
    if Path(build_file).suffix.lower() == ".cfg":
        command = (
            "cd '{workspace}' && "
            f"vitis-run --mode hls --csim --config {quoted} --work_dir vitis_work"
        )
    else:
        command = "cd '{workspace}' && " f"vitis-run --mode hls --tcl {quoted}"
    return ["bash", "-lc", command]


def resolve_benchmark(root: Path) -> TaskManifest:
    """Discover a benchmark and return one in-memory normalised task."""

    benchmark = discover_benchmark(root)
    source = _canonical_source(benchmark)
    root = benchmark.root.resolve()

    source_relative = _root_relative(source, root)
    testbench_relative = [_root_relative(path, root) for path in benchmark.testbenches]
    header_relative = [_root_relative(path, root) for path in benchmark.headers]
    build_relative = _root_relative(benchmark.tcl, root)

    protected_files = [*header_relative, *testbench_relative, build_relative]
    include_dir = str(Path(source_relative).parent)
    host_command = [
        "g++",
        "-std=c++17",
        "-I",
        include_dir,
        source_relative,
        *testbench_relative,
        "-o",
        ".agent_host_test",
    ]

    task_id = f"auto_{benchmark.name}_001"
    data = {
        "task_id": task_id,
        "task_kind": "unknown",
        "task_root": _portable_path(root),
        "artifacts": {
            "source": _portable_path(source),
            "testbench": [_portable_path(path) for path in benchmark.testbenches],
            "headers": [_portable_path(path) for path in benchmark.headers],
            "build_files": [_portable_path(benchmark.tcl)],
        },
        "interface": {
            "top_function": benchmark.top_function,
            "language": "cpp",
            "numerical_tolerance": None,
            "protected_contract": [
                "Preserve the top-level function signature.",
                "Do not modify the supplied testbench or build configuration.",
            ],
        },
        "target": {
            "tool": "AMD Vitis HLS",
            "tool_version": "2025.2",
            "part": benchmark.part,
            "clock_period_ns": benchmark.clock_period_ns,
            "minimum_frequency_mhz": 100.0,
            "resource_limits": {},
        },
        "budgets": dict(DEFAULT_BUDGETS),
        "model": dict(DEFAULT_MODEL),
        "repair": {
            "benchmark_source": _portable_path(root),
            "editable_files": [source_relative],
            "protected_files": protected_files,
            "context_files": [*header_relative, *testbench_relative],
            "host_validation": {
                "command": host_command,
                "run_command": ["./.agent_host_test"],
            },
            "independent_validation": {
                "enabled": True,
                "command": _independent_validation_command(build_relative),
            },
        },
        "optimisation": {
            "prompt_constraints": list(DEFAULT_OPTIMISATION["prompt_constraints"]),
            "validation": dict(DEFAULT_OPTIMISATION["validation"]),
            "selection": dict(DEFAULT_OPTIMISATION["selection"]),
        },
        "adapter": {"kind": "auto"},
        "output_dir": f"experiments/track_a/{task_id}",
    }
    return TaskManifest(path=root, data=data)


def onboard_benchmark(root: Path) -> TaskManifest:
    """Resolve a directory without writing generated configuration files."""

    task = resolve_benchmark(root)
    print("Automatic benchmark discovery")
    print(f"Benchmark: {Path(task.data['task_root']).name}")
    print(f"Build configuration: {task.data['artifacts']['build_files'][0]}")
    print(f"Top function: {task.data['interface']['top_function']}")
    print(f"Source: {task.data['artifacts']['source']}")
    print(f"Testbench files: {len(task.data['artifacts']['testbench'])}")
    print(f"Part: {task.data['target']['part']}")
    print(f"Clock: {task.data['target']['clock_period_ns']:g} ns")
    return task
