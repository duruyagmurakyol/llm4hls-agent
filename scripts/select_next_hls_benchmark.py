#!/usr/bin/env python3

"""Inventory local HLS benchmarks and rank them for the next autonomous PPA run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}


def source_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )


def detect_top_names(text: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(
        r'(?:extern\s+"C"\s+)?(?:void|int|float|double|long|short|unsigned|signed)'
        r'[\w\s:*&<>]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{',
        text,
        re.MULTILINE,
    ):
        name = match.group(1)
        if name not in {"main"} and name not in names:
            names.append(name)
    return names


def analyse_benchmark(directory: Path) -> dict[str, Any]:
    sources = source_files(directory)
    headers = sorted(path for path in directory.rglob("*.h") if path.is_file())
    tcls = sorted(path for path in directory.rglob("*.tcl") if path.is_file())

    testbenches: list[Path] = []
    design_sources: list[Path] = []
    top_names: set[str] = set()
    has_pipeline = False
    has_loop = False

    for path in sources:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lower_name = path.name.lower()
        looks_tb = (
            "test" in lower_name
            or "tb" in lower_name
            or bool(re.search(r"\bint\s+main\s*\(", text))
        )
        if looks_tb:
            testbenches.append(path)
        else:
            design_sources.append(path)
            top_names.update(detect_top_names(text))
            has_pipeline = has_pipeline or "#pragma HLS PIPELINE" in text
            has_loop = has_loop or bool(re.search(r"\b(for|while)\s*\(", text))

    has_set_top = False
    set_top_names: list[str] = []
    has_csim = False
    has_csynth = False
    for path in tcls:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in re.findall(r"\bset_top\s+([A-Za-z_]\w*)", text):
            if name not in set_top_names:
                set_top_names.append(name)
        has_set_top = has_set_top or bool(set_top_names)
        has_csim = has_csim or "csim_design" in text
        has_csynth = has_csynth or "csynth_design" in text

    checks = {
        "has_design_source": bool(design_sources),
        "has_testbench": bool(testbenches),
        "has_tcl": bool(tcls),
        "has_set_top": has_set_top,
        "has_csim_command": has_csim,
        "has_csynth_command": has_csynth,
        "has_loop": has_loop,
    }
    score = sum(1 for value in checks.values() if value)
    ready = all(
        checks[key]
        for key in (
            "has_design_source",
            "has_testbench",
            "has_tcl",
            "has_set_top",
            "has_csim_command",
            "has_csynth_command",
        )
    )

    return {
        "benchmark": directory.name,
        "directory": str(directory.relative_to(REPO_ROOT)),
        "score": score,
        "ready": ready,
        "checks": checks,
        "design_sources": [str(path.relative_to(REPO_ROOT)) for path in design_sources],
        "testbenches": [str(path.relative_to(REPO_ROOT)) for path in testbenches],
        "headers": [str(path.relative_to(REPO_ROOT)) for path in headers],
        "tcl_scripts": [str(path.relative_to(REPO_ROOT)) for path in tcls],
        "set_top_names": set_top_names,
        "detected_function_names": sorted(top_names),
        "already_pipelined": has_pipeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank repository HLS benchmarks by workflow readiness.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmarks/hls_eval"),
        help="Benchmark root relative to the repository (default: benchmarks/hls_eval)",
    )
    parser.add_argument("--exclude", action="append", default=["atax"])
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root not found: {root}")

    excluded = set(args.exclude)
    directories = sorted(path for path in root.iterdir() if path.is_dir() and path.name not in excluded)
    records = [analyse_benchmark(path) for path in directories]
    records.sort(key=lambda item: (-item["score"], item["benchmark"]))

    report = {
        "benchmark_root": str(root.relative_to(REPO_ROOT)),
        "excluded": sorted(excluded),
        "benchmarks_found": len(records),
        "ranked_benchmarks": records,
    }
    output = REPO_ROOT / "experiments" / "next_benchmark_candidates.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nNext HLS benchmark discovery")
    print(f"Root: {root.relative_to(REPO_ROOT)}")
    print(f"Benchmarks found: {len(records)}")
    print("\nTop candidates")
    for record in records[:10]:
        status = "READY" if record["ready"] else "PARTIAL"
        tops = ", ".join(record["set_top_names"]) or "unknown"
        print(
            f"  {record['benchmark']}: {status}, score={record['score']}/7, "
            f"top={tops}, sources={len(record['design_sources'])}, "
            f"testbenches={len(record['testbenches'])}"
        )
    print(f"\nReport: {output.relative_to(REPO_ROOT)}")
    print("No model call, CSim, or synthesis was run.")


if __name__ == "__main__":
    main()
