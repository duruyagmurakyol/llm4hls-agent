#!/usr/bin/env python3

"""Create a lightweight vector-add workspace for PPA experiments."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "benchmarks/vector_add/ppa_baseline"

FILES = {
    "src/vector_add.h": """#ifndef VECTOR_ADD_H\n#define VECTOR_ADD_H\n\nconstexpr int VECTOR_SIZE = 256;\n\nvoid vector_add(const int a[VECTOR_SIZE], const int b[VECTOR_SIZE], int c[VECTOR_SIZE]);\n\n#endif\n""",
    "src/vector_add.cpp": """#include \"vector_add.h\"\n\nvoid vector_add(const int a[VECTOR_SIZE], const int b[VECTOR_SIZE], int c[VECTOR_SIZE]) {\n    for (int i = 0; i < VECTOR_SIZE; ++i) {\n        c[i] = a[i] + b[i];\n    }\n}\n""",
    "testbench/vector_add_test.cpp": """#include <iostream>\n#include \"vector_add.h\"\n\nint main() {\n    int a[VECTOR_SIZE];\n    int b[VECTOR_SIZE];\n    int c[VECTOR_SIZE] = {};\n\n    for (int i = 0; i < VECTOR_SIZE; ++i) {\n        a[i] = (i * 7) - 13;\n        b[i] = (i * 3) + 5;\n    }\n\n    vector_add(a, b, c);\n\n    for (int i = 0; i < VECTOR_SIZE; ++i) {\n        const int expected = a[i] + b[i];\n        if (c[i] != expected) {\n            std::cerr << \"FAIL index=\" << i << \" expected=\" << expected\n                      << \" actual=\" << c[i] << '\\n';\n            return 1;\n        }\n    }\n\n    std::cout << \"All vector-add tests passed.\\n\";\n    return 0;\n}\n""",
    "run_hls.tcl": """open_project -reset vector_add_hls\nset_top vector_add\nadd_files src/vector_add.cpp -cflags \"-Isrc\"\nopen_solution -reset solution1 -flow_target vivado\nset_part {xczu3eg-sfvc784-2-e}\ncreate_clock -period 10 -name default\ncsynth_design\nexit\n""",
    "task.cfg": """Benchmark: vector_add\nGoal: preserve exact integer vector-add behaviour while improving HLS latency and/or resource use.\nEditable file: src/vector_add.cpp\nProtected files: src/vector_add.h, testbench/vector_add_test.cpp, run_hls.tcl\nTarget: xczu3eg-sfvc784-2-e\nClock target: 10 ns\nVector size: 256\nMode: lightweight (host correctness validation plus synthesis only)\n""",
}

CONFIG = {
    "schema_version": 1,
    "experiment_id": "vector_add_ppa_qwen35",
    "provider": "siliconflow",
    "model": "Qwen/Qwen3.5-122B-A10B",
    "workspace": "benchmarks/vector_add/ppa_baseline",
    "editable_file": "src/vector_add.cpp",
    "protected_files": [
        "src/vector_add.h",
        "testbench/vector_add_test.cpp",
        "run_hls.tcl",
        "task.cfg",
    ],
    "temperature": 0.0,
    "max_output_tokens": 1536,
    "thinking_budget": 0,
    "api_timeout_seconds": 180,
    "max_candidates": 1,
    "synthesis_budget": 2,
    "primary_objective": "latency_best_cycles",
    "acceptance": {
        "require_host_pass": True,
        "require_csim_pass": False,
        "require_primary_non_regression": True,
        "require_any_metric_improvement": True,
    },
}


def main() -> None:
    for relative, content in FILES.items():
        path = WORKSPACE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")

    config_path = ROOT / "configs/vector_add_ppa_qwen35.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {config_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
