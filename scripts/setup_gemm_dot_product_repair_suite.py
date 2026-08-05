#!/usr/bin/env python3

"""Create controlled repair cases for GEMM and dot product.

The suite extends repair evaluation beyond vector add, ATAX and BICG while
keeping the same correctness-first contract: only the implementation source is
editable and all testbench/build artefacts are protected.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
PART = "xczu3eg-sfvc784-2-e"
MODEL = "Qwen/Qwen3.5-122B-A10B"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {label} pattern, found {count}: {old!r}"
        )
    return text.replace(old, new, 1)


def gemm_functional(text: str) -> str:
    return replace_once(
        text,
        "C[i][j] += alpha * A[i][k] * B[k][j];",
        "C[i][j] -= alpha * A[i][k] * B[k][j];",
        label="GEMM accumulation",
    )


def gemm_indexing(text: str) -> str:
    return replace_once(
        text,
        "C[i][j] += alpha * A[i][k] * B[k][j];",
        "C[i][j] += alpha * A[i][k] * B[k][(j + 1) % nj];",
        label="GEMM B access",
    )


def gemm_loop_bound(text: str) -> str:
    return replace_once(
        text,
        "for (k = 0; k < nk; k++)",
        "for (k = 0; k < nk - 1; k++)",
        label="GEMM reduction loop",
    )


def gemm_staged(text: str) -> str:
    text = replace_once(
        text,
        "C[i][j] *= beta;",
        "C[i][j] *= beta",
        label="GEMM beta statement",
    )
    return gemm_indexing(text)


def dot_accumulator(text: str) -> str:
    return replace_once(
        text,
        "sum += a[i] * b[i];",
        "sum = a[i] * b[i];",
        label="dot-product accumulation",
    )


def dot_indexing(text: str) -> str:
    return replace_once(
        text,
        "sum += a[i] * b[i];",
        "sum += a[i] * b[(i + 1) % VECTOR_SIZE];",
        label="dot-product input access",
    )


def dot_loop_bound(text: str) -> str:
    return replace_once(
        text,
        "for (int i = 0; i < VECTOR_SIZE; i++)",
        "for (int i = 0; i < VECTOR_SIZE - 1; i++)",
        label="dot-product loop bound",
    )


def dot_staged(text: str) -> str:
    text = replace_once(
        text,
        "int sum = 0;",
        "int sum = 0",
        label="dot-product initialisation",
    )
    return replace_once(
        text,
        "sum += a[i] * b[i];",
        "sum += a[i] * b[(i + 1) % VECTOR_SIZE];",
        label="dot-product staged input access",
    )


def task_cfg(source_name: str, top: str, testbench_name: str) -> str:
    return f"""[hls]
flow_target=vivado
syn.file=src/{source_name}
syn.top={top}
syn.cflags=-Isrc
tb.file=testbench/{testbench_name}
tb.cflags=-Isrc
part={PART}
clock=10ns
"""


def task_manifest(
    *,
    benchmark: str,
    case: str,
    source_name: str,
    header_name: str,
    testbench_name: str,
    top: str,
    contract: list[str],
) -> dict[str, object]:
    task_id = f"{benchmark}_{case}_repair"
    benchmark_source = f"benchmarks/repair_suite/{benchmark}/{case}"
    source = f"{benchmark_source}/src/{source_name}"
    header = f"{benchmark_source}/src/{header_name}"
    testbench = f"{benchmark_source}/testbench/{testbench_name}"
    build_file = f"{benchmark_source}/task.cfg"

    return {
        "task_id": task_id,
        "task_kind": "functional_failure",
        "artifacts": {
            "source": source,
            "testbench": [testbench],
            "headers": [header],
            "specification": None,
            "build_files": [build_file],
        },
        "interface": {
            "top_function": top,
            "language": "cpp",
            "numerical_tolerance": 1e-8 if benchmark == "gemm" else None,
            "protected_contract": [
                f"Preserve the exact {top} function signature.",
                f"Modify only src/{source_name}.",
                *contract,
            ],
        },
        "target": {
            "tool": "AMD Vitis HLS",
            "tool_version": "2025.2",
            "part": PART,
            "clock_period_ns": 10.0,
            "minimum_frequency_mhz": 100.0,
            "resource_limits": {},
        },
        "budgets": {
            "max_iterations": 3,
            "max_csim_calls": 4,
            "max_cosim_calls": 2,
            "max_synthesis_calls": 2,
            "max_model_calls": 3,
            "max_total_tokens": 12288,
        },
        "model": {
            "provider": "siliconflow",
            "name": MODEL,
            "temperature": 0.0,
            "max_tokens": 2048,
            "timeout_seconds": 180,
            "enable_thinking": False,
        },
        "repair": {
            "benchmark_source": benchmark_source,
            "editable_files": [f"src/{source_name}"],
            "protected_files": [
                f"src/{header_name}",
                f"testbench/{testbench_name}",
                "task.cfg",
            ],
            "context_files": [
                f"src/{header_name}",
                f"testbench/{testbench_name}",
                "task.cfg",
            ],
            "host_validation": {
                "command": [
                    "g++",
                    "-std=c++17",
                    "-Isrc",
                    f"src/{source_name}",
                    f"testbench/{testbench_name}",
                    "-o",
                    "host_test",
                ],
                "run_command": ["./host_test"],
            },
            "independent_validation": {
                "enabled": True,
                "command": [
                    "bash",
                    "-lc",
                    "cd '{workspace}' && vitis-run --mode hls --csim --config task.cfg --work_dir vitis_work",
                ],
            },
        },
        "adapter": {"kind": "direct_api_repair"},
        "output_dir": f"runs/repair_suite/{task_id}",
    }


def materialise_case(
    *,
    benchmark: str,
    case: str,
    golden_source: Path,
    golden_header: Path,
    golden_testbench: Path,
    source_name: str,
    header_name: str,
    testbench_name: str,
    top: str,
    transform: Callable[[str], str],
    contract: list[str],
    force: bool,
) -> Path:
    destination = REPO_ROOT / "benchmarks" / "repair_suite" / benchmark / case
    task_path = REPO_ROOT / "configs" / "tasks" / "repair_suite" / f"{benchmark}_{case}.json"

    if destination.exists():
        if not force:
            raise FileExistsError(
                f"Repair case already exists: {destination}. Use --force to regenerate it."
            )
        shutil.rmtree(destination)

    source = golden_source.read_text(encoding="utf-8")
    write_text(destination / "src" / source_name, transform(source))
    write_text(
        destination / "src" / header_name,
        golden_header.read_text(encoding="utf-8"),
    )
    write_text(
        destination / "testbench" / testbench_name,
        golden_testbench.read_text(encoding="utf-8"),
    )
    write_text(destination / "task.cfg", task_cfg(source_name, top, testbench_name))
    write_json(
        task_path,
        task_manifest(
            benchmark=benchmark,
            case=case,
            source_name=source_name,
            header_name=header_name,
            testbench_name=testbench_name,
            top=top,
            contract=contract,
        ),
    )
    return task_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create controlled GEMM and dot-product repair cases."
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    gemm_root = REPO_ROOT / "benchmarks" / "hls_eval" / "gemm"
    dot_root = REPO_ROOT / "benchmarks" / "dot_product"
    required = [
        gemm_root / "src" / "gemm.cpp",
        gemm_root / "src" / "gemm.h",
        gemm_root / "testbench" / "gemm_tb.cpp",
        dot_root / "src" / "dot_product.cpp",
        dot_root / "src" / "dot_product.h",
        dot_root / "testbench" / "dot_product_test.cpp",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark inputs: " + ", ".join(missing))

    gemm_contract = [
        "For every i and j, compute beta times the original C[i][j] plus all 30 alpha*A[i][k]*B[k][j] terms.",
        "Keep every access within C[20][25], A[20][30] and B[30][25].",
    ]
    dot_contract = [
        "Compute the full eight-element dot product exactly once per element.",
        "Write the accumulated result through the result reference.",
    ]

    cases: list[tuple[str, str, Callable[[str], str]]] = [
        ("gemm", "functional_wrong_sign", gemm_functional),
        ("gemm", "indexing_shift_b", gemm_indexing),
        ("gemm", "loop_bound_missing_k", gemm_loop_bound),
        ("gemm", "staged_compile_then_indexing", gemm_staged),
        ("dot_product", "accumulator_overwrite", dot_accumulator),
        ("dot_product", "indexing_shift_b", dot_indexing),
        ("dot_product", "loop_bound_missing_last", dot_loop_bound),
        ("dot_product", "staged_compile_then_indexing", dot_staged),
    ]

    created: list[Path] = []
    for benchmark, case, transform in cases:
        if benchmark == "gemm":
            created.append(
                materialise_case(
                    benchmark=benchmark,
                    case=case,
                    golden_source=gemm_root / "src" / "gemm.cpp",
                    golden_header=gemm_root / "src" / "gemm.h",
                    golden_testbench=gemm_root / "testbench" / "gemm_tb.cpp",
                    source_name="gemm.cpp",
                    header_name="gemm.h",
                    testbench_name="gemm_tb.cpp",
                    top="kernel_gemm",
                    transform=transform,
                    contract=gemm_contract,
                    force=args.force,
                )
            )
        else:
            created.append(
                materialise_case(
                    benchmark=benchmark,
                    case=case,
                    golden_source=dot_root / "src" / "dot_product.cpp",
                    golden_header=dot_root / "src" / "dot_product.h",
                    golden_testbench=dot_root / "testbench" / "dot_product_test.cpp",
                    source_name="dot_product.cpp",
                    header_name="dot_product.h",
                    testbench_name="dot_product_test.cpp",
                    top="dot_product",
                    transform=transform,
                    contract=dot_contract,
                    force=args.force,
                )
            )

    index = {
        "schema_version": 1,
        "benchmarks": ["gemm", "dot_product"],
        "cases": [str(path.relative_to(REPO_ROOT)) for path in created],
        "model": MODEL,
        "repetitions_recommended": 3,
    }
    index_path = REPO_ROOT / "configs" / "tasks" / "repair_suite" / "index.json"
    write_json(index_path, index)

    print("Created controlled repair suite:")
    for path in created:
        print("-", path.relative_to(REPO_ROOT))
    print("Index:", index_path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
