#!/usr/bin/env python3

"""Import selected real HLS-Eval PolyBench kernels and create repair tasks.

The original source, header, description and upstream testbench are retained in
``golden``.  Each repair case receives a deliberately faulty implementation and
a self-checking testbench, because the upstream PolyBench harnesses print arrays
rather than returning a failure when results are incorrect.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REPOSITORY = "sharc-lab/hls-eval"
UPSTREAM_COMMIT = "adea9ff46ab3dea51a8e1790b9d8c4da7899275b"
UPSTREAM_RAW = (
    f"https://raw.githubusercontent.com/{UPSTREAM_REPOSITORY}/"
    f"{UPSTREAM_COMMIT}/hls_eval_data/polybench"
)
PART = "xczu3eg-sfvc784-2-e"
MODEL = "Qwen/Qwen3.5-122B-A10B"


GEMVER_TB = r'''#include <cmath>
#include <cstdio>

#include "gemver.h"

static void init_array(
    double *alpha,
    double *beta,
    double A[40][40],
    double u1[40],
    double v1[40],
    double u2[40],
    double v2[40],
    double w[40],
    double x[40],
    double y[40],
    double z[40]) {
    *alpha = 1.5;
    *beta = 1.2;
    const double n = 40.0;

    for (int i = 0; i < 40; ++i) {
        u1[i] = i;
        u2[i] = ((i + 1) / n) / 2.0;
        v1[i] = ((i + 1) / n) / 4.0;
        v2[i] = ((i + 1) / n) / 6.0;
        y[i] = ((i + 1) / n) / 8.0;
        z[i] = ((i + 1) / n) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (int j = 0; j < 40; ++j)
            A[i][j] = static_cast<double>((i * j + 3 * i + j) % 40) / 40.0;
    }
}

static void reference_gemver(
    double alpha,
    double beta,
    double A[40][40],
    const double u1[40],
    const double v1[40],
    const double u2[40],
    const double v2[40],
    double w[40],
    double x[40],
    const double y[40],
    const double z[40]) {
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            x[i] = x[i] + beta * A[j][i] * y[j];

    for (int i = 0; i < 40; ++i)
        x[i] = x[i] + z[i];

    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            w[i] = w[i] + alpha * A[i][j] * x[j];
}

static bool close(double actual, double expected) {
    return std::fabs(actual - expected) <= 1e-8;
}

int main() {
    double alpha;
    double beta;
    double A[40][40];
    double expected_A[40][40];
    double u1[40];
    double v1[40];
    double u2[40];
    double v2[40];
    double w[40];
    double expected_w[40];
    double x[40];
    double expected_x[40];
    double y[40];
    double z[40];

    init_array(&alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
    for (int i = 0; i < 40; ++i) {
        expected_w[i] = w[i];
        expected_x[i] = x[i];
        for (int j = 0; j < 40; ++j)
            expected_A[i][j] = A[i][j];
    }

    reference_gemver(
        alpha, beta, expected_A, u1, v1, u2, v2,
        expected_w, expected_x, y, z
    );
    kernel_gemver(alpha, beta, A, u1, v1, u2, v2, w, x, y, z);

    for (int i = 0; i < 40; ++i) {
        if (!close(x[i], expected_x[i])) {
            std::fprintf(stderr, "FAIL x[%d]: expected %.12f, got %.12f\n", i, expected_x[i], x[i]);
            return 1;
        }
        if (!close(w[i], expected_w[i])) {
            std::fprintf(stderr, "FAIL w[%d]: expected %.12f, got %.12f\n", i, expected_w[i], w[i]);
            return 1;
        }
        for (int j = 0; j < 40; ++j) {
            if (!close(A[i][j], expected_A[i][j])) {
                std::fprintf(stderr, "FAIL A[%d][%d]\n", i, j);
                return 1;
            }
        }
    }

    std::printf("All GEMVER tests passed.\n");
    return 0;
}
'''


GESUMMV_TB = r'''#include <cmath>
#include <cstdio>

#include "gesummv.h"

static void init_array(
    double *alpha,
    double *beta,
    double A[30][30],
    double B[30][30],
    double x[30]) {
    *alpha = 1.5;
    *beta = 1.2;
    for (int i = 0; i < 30; ++i) {
        x[i] = static_cast<double>(i) / 30.0;
        for (int j = 0; j < 30; ++j) {
            A[i][j] = static_cast<double>((i * j + 1) % 30) / 30.0;
            B[i][j] = static_cast<double>((i * j + 2) % 30) / 30.0;
        }
    }
}

static void reference_gesummv(
    double alpha,
    double beta,
    const double A[30][30],
    const double B[30][30],
    double tmp[30],
    const double x[30],
    double y[30]) {
    for (int i = 0; i < 30; ++i) {
        tmp[i] = 0.0;
        y[i] = 0.0;
        for (int j = 0; j < 30; ++j) {
            tmp[i] = A[i][j] * x[j] + tmp[i];
            y[i] = B[i][j] * x[j] + y[i];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

int main() {
    double alpha;
    double beta;
    double A[30][30];
    double B[30][30];
    double tmp[30];
    double expected_tmp[30];
    double x[30];
    double y[30];
    double expected_y[30];

    init_array(&alpha, &beta, A, B, x);
    reference_gesummv(alpha, beta, A, B, expected_tmp, x, expected_y);
    kernel_gesummv(alpha, beta, A, B, tmp, x, y);

    for (int i = 0; i < 30; ++i) {
        if (std::fabs(tmp[i] - expected_tmp[i]) > 1e-8) {
            std::fprintf(stderr, "FAIL tmp[%d]: expected %.12f, got %.12f\n", i, expected_tmp[i], tmp[i]);
            return 1;
        }
        if (std::fabs(y[i] - expected_y[i]) > 1e-8) {
            std::fprintf(stderr, "FAIL y[%d]: expected %.12f, got %.12f\n", i, expected_y[i], y[i]);
            return 1;
        }
    }

    std::printf("All GESUMMV tests passed.\n");
    return 0;
}
'''


MVT_TB = r'''#include <cmath>
#include <cstdio>

#include "mvt.h"

static void init_array(
    double x1[40],
    double x2[40],
    double y1[40],
    double y2[40],
    double A[40][40]) {
    for (int i = 0; i < 40; ++i) {
        x1[i] = static_cast<double>(i) / 40.0;
        x2[i] = static_cast<double>(i + 1) / 40.0;
        y1[i] = static_cast<double>(i + 3) / 40.0;
        y2[i] = static_cast<double>(i + 4) / 40.0;
        for (int j = 0; j < 40; ++j)
            A[i][j] = static_cast<double>((i * (j + 1) + 2 * j + 1) % 40) / 40.0;
    }
}

static void reference_mvt(
    double x1[40],
    double x2[40],
    const double y1[40],
    const double y2[40],
    const double A[40][40]) {
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            x1[i] = x1[i] + A[i][j] * y1[j];

    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            x2[i] = x2[i] + A[j][i] * y2[j];
}

int main() {
    double x1[40];
    double expected_x1[40];
    double x2[40];
    double expected_x2[40];
    double y1[40];
    double y2[40];
    double A[40][40];

    init_array(x1, x2, y1, y2, A);
    for (int i = 0; i < 40; ++i) {
        expected_x1[i] = x1[i];
        expected_x2[i] = x2[i];
    }

    reference_mvt(expected_x1, expected_x2, y1, y2, A);
    kernel_mvt(x1, x2, y1, y2, A);

    for (int i = 0; i < 40; ++i) {
        if (std::fabs(x1[i] - expected_x1[i]) > 1e-8) {
            std::fprintf(stderr, "FAIL x1[%d]: expected %.12f, got %.12f\n", i, expected_x1[i], x1[i]);
            return 1;
        }
        if (std::fabs(x2[i] - expected_x2[i]) > 1e-8) {
            std::fprintf(stderr, "FAIL x2[%d]: expected %.12f, got %.12f\n", i, expected_x2[i], x2[i]);
            return 1;
        }
    }

    std::printf("All MVT tests passed.\n");
    return 0;
}
'''


SYRK_TB = r'''#include <cmath>
#include <cstdio>

#include "syrk.h"

static void init_array(
    double *alpha,
    double *beta,
    double C[30][30],
    double A[30][20]) {
    *alpha = 1.5;
    *beta = 1.2;
    for (int i = 0; i < 30; ++i)
        for (int j = 0; j < 20; ++j)
            A[i][j] = static_cast<double>((i * j + 1) % 30) / 30.0;
    for (int i = 0; i < 30; ++i)
        for (int j = 0; j < 30; ++j)
            C[i][j] = static_cast<double>((i * j + 2) % 20) / 20.0;
}

static void reference_syrk(
    double alpha,
    double beta,
    double C[30][30],
    const double A[30][20]) {
    for (int i = 0; i < 30; ++i) {
        for (int j = 0; j <= i; ++j)
            C[i][j] *= beta;
        for (int k = 0; k < 20; ++k)
            for (int j = 0; j <= i; ++j)
                C[i][j] += alpha * A[i][k] * A[j][k];
    }
}

int main() {
    double alpha;
    double beta;
    double C[30][30];
    double expected_C[30][30];
    double A[30][20];

    init_array(&alpha, &beta, C, A);
    for (int i = 0; i < 30; ++i)
        for (int j = 0; j < 30; ++j)
            expected_C[i][j] = C[i][j];

    reference_syrk(alpha, beta, expected_C, A);
    kernel_syrk(alpha, beta, C, A);

    for (int i = 0; i < 30; ++i) {
        for (int j = 0; j < 30; ++j) {
            if (std::fabs(C[i][j] - expected_C[i][j]) > 1e-8) {
                std::fprintf(stderr, "FAIL C[%d][%d]: expected %.12f, got %.12f\n", i, j, expected_C[i][j], C[i][j]);
                return 1;
            }
        }
    }

    std::printf("All SYRK tests passed.\n");
    return 0;
}
'''


class Benchmark:
    def __init__(
        self,
        name: str,
        top: str,
        fault_name: str,
        fault: Callable[[str], str],
        testbench: str,
        contract: list[str],
    ) -> None:
        self.name = name
        self.top = top
        self.fault_name = fault_name
        self.fault = fault
        self.testbench = testbench
        self.contract = contract


def replace_once(text: str, old: str, new: str, *, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one {description} pattern, found {count}: {old!r}"
        )
    return text.replace(old, new, 1)


def fault_gemver(text: str) -> str:
    return replace_once(
        text,
        "x[i] = x[i] + beta * A[j][i] * y[j];",
        "x[i] = x[i] + beta * A[i][j] * y[j];",
        description="GEMVER transpose access",
    )


def fault_gesummv(text: str) -> str:
    return replace_once(
        text,
        "tmp[i] = A[i][j] * x[j] + tmp[i];",
        "tmp[i] = A[i][j] * x[j];",
        description="GESUMMV accumulation",
    )


def fault_mvt(text: str) -> str:
    return replace_once(
        text,
        "x2[i] = x2[i] + A[j][i] * y_2[j];",
        "x2[i] = x2[i] + A[j][i] * y_2[(j + 1) % n];",
        description="MVT second-vector index",
    )


def fault_syrk(text: str) -> str:
    return replace_once(
        text,
        "for (j = 0; j <= i; j++)\n            C[i][j] *= beta;",
        "for (j = 0; j < i; j++)\n            C[i][j] *= beta;",
        description="SYRK diagonal scaling loop",
    )


BENCHMARKS = [
    Benchmark(
        "gemver",
        "kernel_gemver",
        "transpose_access",
        fault_gemver,
        GEMVER_TB,
        [
            "Perform both rank-1 updates of A before using A in later stages.",
            "Use the transposed access A[j][i] when accumulating x.",
            "Add z to x and then compute w from all 40 columns.",
        ],
    ),
    Benchmark(
        "gesummv",
        "kernel_gesummv",
        "accumulator_overwrite",
        fault_gesummv,
        GESUMMV_TB,
        [
            "Accumulate all 30 products into tmp[i] and y[i].",
            "Compute final y[i] as alpha*tmp[i] + beta*y[i].",
        ],
    ),
    Benchmark(
        "mvt",
        "kernel_mvt",
        "shifted_second_vector",
        fault_mvt,
        MVT_TB,
        [
            "Update x1 using A[i][j] and y_1[j].",
            "Update x2 using A[j][i] and the matching y_2[j] element.",
        ],
    ),
    Benchmark(
        "syrk",
        "kernel_syrk",
        "missing_diagonal_scaling",
        fault_syrk,
        SYRK_TB,
        [
            "Scale every lower-triangular element including C[i][i] by beta.",
            "Accumulate all 20 alpha*A[i][k]*A[j][k] terms for j <= i.",
            "Leave the upper triangle unchanged.",
        ],
    ),
]


def fetch_text(benchmark: str, filename: str) -> str:
    url = f"{UPSTREAM_RAW}/{benchmark}/{filename}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llm4hls-agent-benchmark-importer"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not download {url}: {error}") from error


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def task_cfg(benchmark: Benchmark) -> str:
    return f"""[hls]
flow_target=vivado
syn.file=src/{benchmark.name}.cpp
syn.top={benchmark.top}
syn.cflags=-Isrc
tb.file=testbench/{benchmark.name}_tb.cpp
tb.cflags=-Isrc
part={PART}
clock=10ns
"""


def manifest(benchmark: Benchmark, fault_root: Path) -> dict[str, object]:
    task_id = f"hls_eval_{benchmark.name}_{benchmark.fault_name}_repair"
    source = fault_root / "src" / f"{benchmark.name}.cpp"
    header = fault_root / "src" / f"{benchmark.name}.h"
    testbench = fault_root / "testbench" / f"{benchmark.name}_tb.cpp"
    description = fault_root / "kernel_description.md"
    config = fault_root / "task.cfg"

    return {
        "task_id": task_id,
        "task_kind": "functional_failure",
        "artifacts": {
            "source": relative(source),
            "testbench": [relative(testbench)],
            "headers": [relative(header)],
            "specification": relative(description),
            "build_files": [relative(config)],
        },
        "interface": {
            "top_function": benchmark.top,
            "language": "cpp",
            "numerical_tolerance": 1e-8,
            "protected_contract": [
                f"Preserve the exact {benchmark.top} function signature.",
                f"Modify only src/{benchmark.name}.cpp.",
                *benchmark.contract,
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
            "benchmark_source": relative(fault_root),
            "editable_files": [f"src/{benchmark.name}.cpp"],
            "protected_files": [
                f"src/{benchmark.name}.h",
                f"testbench/{benchmark.name}_tb.cpp",
                "kernel_description.md",
                "task.cfg",
            ],
            "context_files": [
                f"src/{benchmark.name}.h",
                f"testbench/{benchmark.name}_tb.cpp",
                "kernel_description.md",
                "task.cfg",
            ],
            "host_validation": {
                "command": [
                    "g++",
                    "-std=c++17",
                    "-Isrc",
                    f"src/{benchmark.name}.cpp",
                    f"testbench/{benchmark.name}_tb.cpp",
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
        "output_dir": f"runs/hls_eval_imported/{task_id}",
        "provenance": {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_directory": f"hls_eval_data/polybench/{benchmark.name}",
            "testbench_note": (
                "The upstream print-only harness is retained under golden; "
                "the repair case uses a self-checking derivative."
            ),
        },
    }


def materialise(benchmark: Benchmark, *, force: bool) -> Path:
    root = REPO_ROOT / "benchmarks" / "hls_eval_imported" / benchmark.name
    golden = root / "golden"
    fault_root = root / "faults" / benchmark.fault_name
    task_path = (
        REPO_ROOT
        / "configs"
        / "tasks"
        / "hls_eval_imported"
        / f"{benchmark.name}_{benchmark.fault_name}.json"
    )

    if root.exists():
        if not force:
            raise FileExistsError(
                f"Benchmark already exists: {root}. Use --force to regenerate it."
            )
        shutil.rmtree(root)

    source = fetch_text(benchmark.name, f"{benchmark.name}.cpp")
    header = fetch_text(benchmark.name, f"{benchmark.name}.h")
    upstream_tb = fetch_text(benchmark.name, f"{benchmark.name}_tb.cpp")
    description = fetch_text(benchmark.name, "kernel_description.md")
    top = fetch_text(benchmark.name, "top.txt")

    write_text(golden / "src" / f"{benchmark.name}.cpp", source)
    write_text(golden / "src" / f"{benchmark.name}.h", header)
    write_text(golden / "testbench" / f"{benchmark.name}_tb.cpp", upstream_tb)
    write_text(golden / "kernel_description.md", description)
    write_text(golden / "top.txt", top)
    write_json(
        golden / "upstream.json",
        {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "directory": f"hls_eval_data/polybench/{benchmark.name}",
        },
    )

    write_text(
        fault_root / "src" / f"{benchmark.name}.cpp",
        benchmark.fault(source),
    )
    write_text(fault_root / "src" / f"{benchmark.name}.h", header)
    write_text(
        fault_root / "testbench" / f"{benchmark.name}_tb.cpp",
        benchmark.testbench,
    )
    write_text(fault_root / "kernel_description.md", description)
    write_text(fault_root / "task.cfg", task_cfg(benchmark))
    write_json(
        fault_root / "fault.json",
        {
            "benchmark": benchmark.name,
            "fault": benchmark.fault_name,
            "editable_file": f"src/{benchmark.name}.cpp",
            "upstream_commit": UPSTREAM_COMMIT,
        },
    )
    write_json(task_path, manifest(benchmark, fault_root))
    return task_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import selected HLS-Eval kernels as controlled repair tasks."
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    task_paths = [materialise(benchmark, force=args.force) for benchmark in BENCHMARKS]
    index_path = (
        REPO_ROOT / "configs" / "tasks" / "hls_eval_imported" / "index.json"
    )
    write_json(
        index_path,
        {
            "schema_version": 1,
            "source": {
                "repository": UPSTREAM_REPOSITORY,
                "commit": UPSTREAM_COMMIT,
            },
            "benchmarks": [benchmark.name for benchmark in BENCHMARKS],
            "cases": [relative(path) for path in task_paths],
            "model": MODEL,
            "repetitions_recommended": 3,
        },
    )

    print("Imported real HLS-Eval repair benchmarks:")
    for path in task_paths:
        print("-", relative(path))
    print("Index:", relative(index_path))


if __name__ == "__main__":
    main()
