#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "benchmarks/bicg/golden"
FAULT_ROOT = ROOT / "benchmarks/bicg/faults"
CONFIG_ROOT = ROOT / "configs/bicg_api_qwen35"

FAULTS = {
    "functional": (
        "functional_wrong_operator",
        "s[j] = s[j] + r[i] * A[i][j];",
        "s[j] = s[j] - r[i] * A[i][j];",
    ),
    "indexing": (
        "indexing_off_by_one",
        "q[i] = q[i] + A[i][j] * p[j];",
        "q[i] = q[i] + A[i][j] * p[(j + 1) % m];",
    ),
    "interface": (
        "interface_wrong_top_name",
        "void kernel_bicg(",
        "void kernel_bicg_wrong(",
    ),
    "syntax": (
        "syntax_missing_semicolon",
        "q[i] = 0.0;",
        "q[i] = 0.0",
    ),
}


def make_config(name: str, fault_dir: str) -> dict[str, object]:
    return {
        "experiment_id": f"bicg_api_qwen35_{name}",
        "benchmark_source": f"benchmarks/bicg/faults/{fault_dir}",
        "repair_mode": "direct_api",
        "provider": "siliconflow",
        "model": "Qwen/Qwen3.5-122B-A10B",
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "api_timeout_seconds": 180,
        "editable_files": ["src/bicg.cpp"],
        "protected_files": ["src/bicg.h", "testbench/bicg_test.cpp", "task.cfg"],
        "context_files": ["src/bicg.h", "testbench/bicg_test.cpp", "task.cfg"],
        "host_validation": {
            "command": [
                "g++", "-std=c++17", "-Isrc", "src/bicg.cpp",
                "testbench/bicg_test.cpp", "-o", "host_test"
            ],
            "run_command": ["./host_test"],
        },
        "independent_validation": {
            "enabled": True,
            "command": [
                "bash", "-lc",
                "python3 agent/run_repair.py '{workspace}' | tee /dev/stderr | grep -q 'Original result: PASSED'"
            ],
        },
    }


def main() -> None:
    if not GOLDEN.is_dir():
        raise SystemExit(f"Golden BICG benchmark not found: {GOLDEN}")

    FAULT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    golden_source = (GOLDEN / "src/bicg.cpp").read_text(encoding="utf-8")

    for name, (fault_dir, old, new) in FAULTS.items():
        destination = FAULT_ROOT / fault_dir
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(GOLDEN, destination)

        if old not in golden_source:
            raise SystemExit(f"Fault anchor not found for {name}: {old}")

        faulty_source = golden_source.replace(old, new, 1)
        (destination / "src/bicg.cpp").write_text(faulty_source, encoding="utf-8")

        config = make_config(name, fault_dir)
        (CONFIG_ROOT / f"{name}.json").write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )

    print("Created BICG golden-derived fault suite:")
    for name, (fault_dir, _, _) in FAULTS.items():
        print(f"  {name}: benchmarks/bicg/faults/{fault_dir}")
    print(f"Configs: {CONFIG_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
