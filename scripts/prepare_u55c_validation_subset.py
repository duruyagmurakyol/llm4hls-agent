#!/usr/bin/env python3

"""Generate a representative Alveo U55C validation suite from frozen tasks."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shlex
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
U55C_PART = "xcu55c-fsvh2892-2L-e"
U55C_RESOURCE_LIMITS = {
    "lut": 1_303_680,
    "ff": 2_607_360,
    "dsp": 9_024,
    "bram_18k": 4_032,
}
BASE_MANIFESTS = (
    Path("configs/tasks/combined_full_agent/dot_product_accumulator_overwrite_repair_full_agent.json"),
    Path("configs/tasks/combined_full_agent/gemm_loop_bound_missing_k_repair_full_agent.json"),
    Path("configs/tasks/combined_full_agent/hls_eval_gesummv_accumulator_overwrite_repair_full_agent.json"),
    Path("configs/tasks/combined_full_agent/hls_eval_mvt_shifted_second_vector_repair_full_agent.json"),
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _repo_relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _relative_from(path: Path, directory: Path) -> str:
    return Path(os.path.relpath(path.resolve(), directory.resolve())).as_posix()


def _rewrite_include_flags(value: str, source_dir: Path, output_dir: Path) -> str:
    rewritten: list[str] = []
    for token in shlex.split(value):
        if token.startswith("-I") and len(token) > 2:
            include = Path(token[2:])
            if not include.is_absolute():
                include = (source_dir / include).resolve()
            token = "-I" + _relative_from(include, output_dir)
        rewritten.append(token)
    return shlex.join(rewritten)


def _write_target_cfg(
    *,
    source_cfg: Path,
    output_cfg: Path,
    manifest: dict[str, Any],
    repo_root: Path,
) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    if not parser.read(source_cfg, encoding="utf-8") or "hls" not in parser:
        raise ValueError(f"Could not read [hls] configuration from {source_cfg}")

    hls = parser["hls"]
    artifacts = manifest["artifacts"]
    source = repo_root / str(artifacts["source"])
    testbenches = artifacts.get("testbench") or []
    if not source.is_file() or not testbenches:
        raise ValueError(f"Invalid source or testbench in {manifest['task_id']}")
    testbench = repo_root / str(testbenches[0])
    if not testbench.is_file():
        raise FileNotFoundError(f"Testbench not found: {testbench}")

    output_cfg.parent.mkdir(parents=True, exist_ok=True)
    for key in ("syn.cflags", "tb.cflags"):
        if hls.get(key, "").strip():
            hls[key] = _rewrite_include_flags(
                hls[key],
                source_cfg.parent,
                output_cfg.parent,
            )

    hls["flow_target"] = hls.get("flow_target", "vivado")
    hls["syn.file"] = _relative_from(source, output_cfg.parent)
    hls["syn.top"] = str(manifest["interface"]["top_function"])
    hls["tb.file"] = _relative_from(testbench, output_cfg.parent)
    hls["part"] = U55C_PART
    hls["clock"] = "10ns"

    with output_cfg.open("w", encoding="utf-8") as handle:
        parser.write(handle, space_around_delimiters=False)


def _u55c_manifest(
    *,
    source_manifest: Path,
    output_root: Path,
    repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    manifest = _load_json(source_manifest)
    base_id = str(manifest["task_id"])
    task_id = f"{base_id}_u55c"

    source_build_files = manifest["artifacts"].get("build_files") or []
    if len(source_build_files) != 1:
        raise ValueError(f"{base_id} must define exactly one build file")
    source_cfg = repo_root / str(source_build_files[0])
    if not source_cfg.is_file():
        raise FileNotFoundError(f"Build configuration not found: {source_cfg}")

    target_cfg = output_root / "build" / f"{task_id}.cfg"
    _write_target_cfg(
        source_cfg=source_cfg,
        output_cfg=target_cfg,
        manifest=manifest,
        repo_root=repo_root,
    )

    manifest["task_id"] = task_id
    manifest["parent_task_id"] = base_id
    manifest["artifacts"]["build_files"] = [
        _repo_relative(target_cfg, repo_root)
    ]
    manifest["target"] = {
        "tool": "AMD Vitis HLS",
        "tool_version": "2025.2",
        "platform": "Alveo U55C",
        "part": U55C_PART,
        "clock_period_ns": 10.0,
        "minimum_frequency_mhz": 100.0,
        "resource_limits": dict(U55C_RESOURCE_LIMITS),
    }
    manifest["output_dir"] = f"runs/u55c_validation/{task_id}"
    manifest["target_validation"] = {
        "purpose": "competition_target_representative_rerun",
        "source_manifest": _repo_relative(source_manifest, repo_root),
        "target_board": "AMD Alveo U55C",
        "target_part": U55C_PART,
        "clock_period_ns": 10.0,
        "minimum_frequency_mhz": 100.0,
    }

    optimisation = manifest.get("optimisation")
    if isinstance(optimisation, dict):
        constraints = optimisation.get("prompt_constraints")
        if not isinstance(constraints, list):
            constraints = []
        constraints = [
            item
            for item in constraints
            if not (
                isinstance(item, str)
                and item.startswith("Treat 100 MHz and the configured FPGA resource limits")
            )
        ]
        constraints.append(
            "Target the AMD Alveo U55C part xcu55c-fsvh2892-2L-e; treat 100 MHz and the configured U55C resource limits as hard constraints."
        )
        optimisation["prompt_constraints"] = constraints

    output_manifest = output_root / f"{task_id}.json"
    _write_json(output_manifest, manifest)
    return output_manifest, manifest


def prepare_subset(
    *,
    repo_root: Path = REPO_ROOT,
    base_manifests: Iterable[Path] = BASE_MANIFESTS,
    output_root: Path | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    output_root = (
        output_root.resolve()
        if output_root is not None
        else repo_root / "configs/tasks/u55c_validation"
    )

    generated: list[Path] = []
    task_ids: list[str] = []
    for relative_manifest in base_manifests:
        source_manifest = (
            relative_manifest
            if relative_manifest.is_absolute()
            else repo_root / relative_manifest
        )
        output_manifest, manifest = _u55c_manifest(
            source_manifest=source_manifest,
            output_root=output_root,
            repo_root=repo_root,
        )
        generated.append(output_manifest)
        task_ids.append(str(manifest["task_id"]))

    index = {
        "schema_version": 1,
        "suite_kind": "u55c_representative_validation",
        "purpose": "Confirm repair, synthesis, C/RTL co-simulation and PPA selection on the competition target.",
        "cases": [_repo_relative(path, repo_root) for path in generated],
        "task_ids": task_ids,
        "case_count": len(generated),
        "recommended_repetitions": 1,
        "target": {
            "board": "AMD Alveo U55C",
            "part": U55C_PART,
            "tool_version": "2025.2",
            "clock_period_ns": 10.0,
            "minimum_frequency_mhz": 100.0,
            "resource_limits": dict(U55C_RESOURCE_LIMITS),
        },
        "selection_rationale": {
            "dot_product": "repeatable successful latency trade-off",
            "gemm": "conservative baseline fallback",
            "gesummv": "successful imported HLS-Eval optimisation",
            "mvt": "difficult imported case with no verified optimisation candidate in the reference sweep",
        },
    }
    index_path = output_root / "index.json"
    _write_json(index_path, index)
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate target-specific manifests for the U55C representative rerun."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "configs/tasks/u55c_validation",
    )
    args = parser.parse_args()

    index = prepare_subset(output_root=args.output_root)
    print(f"Generated U55C validation suite: {_repo_relative(index, REPO_ROOT)}")
    print(f"Target part: {U55C_PART}")
    print("Cases: 4; recommended repetitions: 1")


if __name__ == "__main__":
    main()
