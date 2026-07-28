#!/usr/bin/env python3

"""Create staged ATAX faults and Qwen3.5 iterative repair configs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "benchmarks/atax/golden"
FAULT_ROOT = ROOT / "benchmarks/atax/faults"
CONFIG_ROOT = ROOT / "configs/atax_iterative_qwen35"

CASES = {
    "staged_compile_then_functional": [
        ("tmp[i] = 0.0;", "tmp[i] = 0.0"),
        ("A[i][j] * x[j]", "A[i][j] * x[(j + 1) % n]"),
    ],
    "staged_interface_then_functional": [
        ("void kernel_atax(", "void kernel_atax_wrong("),
        ("y[j] = y[j] + A[i][j] * tmp[i];", "y[j] = y[j] - A[i][j] * tmp[i];"),
    ],
    "staged_compile_compile_functional": [
        ("y[i] = 0;", "y[i] = 0"),
        ("tmp[i] = 0.0;", "tmp[i] = 0.0"),
        ("A[i][j] * x[j]", "A[i][j] * x[(j + i) % n]"),
    ],
}


def config_for(case: str) -> dict[str, object]:
    return {
        "experiment_id": f"atax_iterative_qwen35_{case}",
        "repair_mode": "iterative_direct_api",
        "provider": "siliconflow",
        "model": "Qwen/Qwen3.5-122B-A10B",
        "benchmark_source": f"benchmarks/atax/faults/{case}",
        "editable_files": ["src/atax.cpp"],
        "protected_files": ["src/atax.h", "testbench/atax_test.cpp", "task.cfg"],
        "context_files": ["src/atax.h", "task.cfg"],
        "temperature": 0.0,
        "max_output_tokens": 2048,
        "api_timeout_seconds": 120,
        "thinking_budget": 0,
        "max_iterations": 3,
        "host_validation": {
            "command": [
                "g++", "-std=c++17", "-Isrc", "src/atax.cpp",
                "testbench/atax_test.cpp", "-o", "host_test"
            ],
            "run_command": ["./host_test"],
        },
        "independent_validation": {
            "enabled": True,
            "command": ["python3", "agent/run_repair.py", "{workspace}"],
        },
    }


def main() -> None:
    if not GOLDEN.is_dir():
        raise SystemExit(f"Missing golden ATAX benchmark: {GOLDEN}")

    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    for case, replacements in CASES.items():
        destination = FAULT_ROOT / case
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(GOLDEN, destination)

        source_path = destination / "src/atax.cpp"
        source = source_path.read_text(encoding="utf-8")
        for old, new in replacements:
            if old not in source:
                raise RuntimeError(f"Pattern not found for {case}: {old}")
            source = source.replace(old, new, 1)
        source_path.write_text(source, encoding="utf-8")

        (destination / "fault.txt").write_text(
            f"Staged ATAX fault for feedback ablation: {case}\n",
            encoding="utf-8",
        )
        (CONFIG_ROOT / f"{case}.json").write_text(
            json.dumps(config_for(case), indent=2) + "\n",
            encoding="utf-8",
        )

    print("Created three staged ATAX faults and Qwen3.5 configs.")


if __name__ == "__main__":
    main()
