#!/usr/bin/env python3

"""Create staged BICG faults and configs for one-shot versus feedback ablation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "benchmarks/bicg/golden"
FAULT_ROOT = ROOT / "benchmarks/bicg/faults"
CONFIG_ROOT = ROOT / "configs/bicg_iterative_qwen35"


def write_fault(name: str, source: str, description: str) -> None:
    target = FAULT_ROOT / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(GOLDEN, target)
    (target / "src/bicg.cpp").write_text(source, encoding="utf-8")
    (target / "fault.txt").write_text(description.strip() + "\n", encoding="utf-8")


def write_config(name: str) -> None:
    config = {
        "experiment_id": f"bicg_iterative_qwen35_{name}",
        "repair_mode": "iterative_direct_api",
        "provider": "siliconflow",
        "model": "Qwen/Qwen3.5-122B-A10B",
        "benchmark_source": f"benchmarks/bicg/faults/{name}",
        "editable_files": ["src/bicg.cpp"],
        "protected_files": ["src/bicg.h", "testbench/bicg_test.cpp", "task.cfg"],
        "context_files": ["src/bicg.h", "task.cfg"],
        "temperature": 0.0,
        "max_output_tokens": 2048,
        "api_timeout_seconds": 120,
        "thinking_budget": 0,
        "max_iterations": 3,
        "host_validation": {
            "command": [
                "g++", "-std=c++17", "-Isrc", "src/bicg.cpp",
                "testbench/bicg_test.cpp", "-o", "host_test"
            ],
            "run_command": ["./host_test"],
        },
        "independent_validation": {
            "enabled": True,
            "command": ["python3", "agent/run_repair.py", "{workspace}"],
        },
    }
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    (CONFIG_ROOT / f"{name}.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    interface_then_functional = '''#include "bicg.h"

void kernel_bicg_wrong(
    double A[42][38],
    double s[38],
    double q[42],
    double p[38],
    double r[42]) {
#pragma HLS top name = kernel_bicg_wrong

    const int n = 42;
    const int m = 38;

    int i, j;

    for (i = 0; i < m; i++)
        s[i] = 0;
    for (i = 0; i < n; i++) {
        q[i] = 0.0;
        for (j = 0; j < m; j++) {
            s[j] = s[j] + r[i] * A[i][j];
            q[i] = q[i] - A[i][j] * p[j];
        }
    }
}
'''

    compile_compile_functional = '''#include "bicg.h"

void kernel_bicg(
    double A[42][38],
    double s[38],
    double q[42],
    double p[38],
    double r[42]) {
#pragma HLS top name = kernel_bicg

    const int n = 42
    const int m = 38;

    int i, j;

    for (i = 0; i < m; i++)
        s[i] = 1.0;
    for (i = 0; i < n; i++) {
        q[i] = 0.0;
        for (j = 0; j < m; j++) {
            s[j] = s[j] + r[i] * A[i][j];
            q[i] = q[i] + A[i][j] * shifted_p[j];
        }
    }
}
'''

    write_fault(
        "staged_interface_then_functional",
        interface_then_functional,
        "Wrong top function name masks a subtraction fault in q accumulation.",
    )
    write_fault(
        "staged_compile_compile_functional",
        compile_compile_functional,
        "Missing semicolon and undeclared shifted_p mask incorrect s initialisation.",
    )
    write_config("staged_interface_then_functional")
    write_config("staged_compile_compile_functional")

    print("Created two staged BICG faults and their iterative configs.")


if __name__ == "__main__":
    main()
