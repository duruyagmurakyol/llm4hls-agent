#!/usr/bin/env python3

"""Combine repair suites into full repair-to-PPA agent tasks.

The source task manifests are kept unchanged. This script writes derived tasks
that use the unified ``auto`` adapter, so every faulty design follows:

    initial validation -> bounded repair -> verified baseline -> PPA search

The combined index currently includes the controlled GEMM/dot-product cases and
the imported real HLS-Eval PolyBench cases.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEXES = (
    Path("configs/tasks/repair_suite/index.json"),
    Path("configs/tasks/hls_eval_imported/index.json"),
)
OUTPUT_ROOT = Path("configs/tasks/combined_full_agent")
RESOURCE_LIMITS = {
    "lut": 70560,
    "ff": 141120,
    "dsp": 360,
    "bram_18k": 432,
}


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in value
    ).strip("_")
    if not result:
        raise ValueError(f"Could not create a safe task name from {value!r}")
    return result


def source_manifests(index_paths: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    seen: set[Path] = set()

    for index_path in index_paths:
        index = load_json(index_path)
        cases = index.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"No cases declared in {index_path}")

        for item in cases:
            manifest = resolve(str(item)).resolve()
            if not manifest.is_file():
                raise FileNotFoundError(f"Task manifest not found: {manifest}")
            if manifest not in seen:
                manifests.append(manifest)
                seen.add(manifest)

    return manifests


def full_agent_task(source_path: Path) -> dict[str, Any]:
    source = load_json(source_path)
    source_id = safe_name(str(source["task_id"]))
    task_id = f"{source_id}_full_agent"
    task = json.loads(json.dumps(source))

    task["task_id"] = task_id
    task["task_kind"] = "unknown"
    task["source_task_manifest"] = str(source_path.relative_to(REPO_ROOT))
    task["adapter"] = {"kind": "auto"}
    task["output_dir"] = f"runs/combined_full_agent/{task_id}"

    target = task.setdefault("target", {})
    target["tool"] = "AMD Vitis HLS"
    target["tool_version"] = "2025.2"
    target["part"] = "xczu3eg-sfvc784-2-e"
    target["clock_period_ns"] = 10.0
    target["minimum_frequency_mhz"] = 100.0
    target["resource_limits"] = dict(RESOURCE_LIMITS)

    # Shared total budget for repair and optimisation. A typical successful run
    # spends one model call on repair and leaves seven calls for PPA candidates.
    # Harder repairs may consume up to three attempts and still leave five PPA
    # candidates. Tool budgets include initial validation and baseline promotion.
    task["budgets"] = {
        "max_iterations": 8,
        "max_csim_calls": 10,
        "max_cosim_calls": 10,
        "max_synthesis_calls": 10,
        "max_model_calls": 8,
        "max_total_tokens": 32768,
    }

    model = task.setdefault("model", {})
    model["provider"] = "siliconflow"
    model["name"] = "Qwen/Qwen3.5-122B-A10B"
    model["temperature"] = 0.0
    model["max_tokens"] = 2048
    model["timeout_seconds"] = 180
    model["enable_thinking"] = False

    interface = task.get("interface") or {}
    top = str(interface.get("top_function", "the top function"))
    existing_optimisation = task.get("optimisation")
    optimisation = (
        existing_optimisation
        if isinstance(existing_optimisation, dict)
        else {}
    )
    validation = optimisation.get("validation")
    if not isinstance(validation, dict):
        validation = {}
    optimisation["validation"] = validation

    configured = optimisation.get("prompt_constraints")
    prompt_constraints = (
        [str(item) for item in configured]
        if isinstance(configured, list)
        else []
    )
    additions = [
        f"Optimise only after {top} has become a fully verified repaired baseline.",
        "Preserve the exact top-level interface and all testbench-observed behaviour.",
        "Treat 100 MHz and the configured FPGA resource limits as hard constraints.",
        "Use hardware-relevant HLS transformations rather than comments, renaming, or no-op rewrites.",
        "Do not completely partition top-level interface arrays.",
    ]
    for item in additions:
        if item not in prompt_constraints:
            prompt_constraints.append(item)
    optimisation["prompt_constraints"] = prompt_constraints
    task["optimisation"] = optimisation

    return task


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one combined repair-to-PPA benchmark suite."
    )
    parser.add_argument(
        "--index",
        action="append",
        default=[],
        help="Source suite index. May be supplied more than once.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    index_paths = [
        resolve(item) for item in (args.index or [str(path) for path in DEFAULT_INDEXES])
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in index_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing source suite indexes: "
            + ", ".join(missing)
            + ". Run the two benchmark setup/import scripts first."
        )

    output_root = resolve(OUTPUT_ROOT)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Combined suite already exists: {output_root}. Use --force to regenerate it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    source_paths = source_manifests(index_paths)
    generated: list[str] = []
    task_ids: list[str] = []

    for source_path in source_paths:
        task = full_agent_task(source_path)
        destination = output_root / f"{task['task_id']}.json"
        write_json(destination, task)
        generated.append(str(destination.relative_to(REPO_ROOT)))
        task_ids.append(str(task["task_id"]))

    index = {
        "schema_version": 1,
        "suite_kind": "repair_then_optimise",
        "source_indexes": [str(path.relative_to(REPO_ROOT)) for path in index_paths],
        "cases": generated,
        "task_ids": task_ids,
        "case_count": len(generated),
        "recommended_repetitions": 3,
        "workflow": [
            "initial_csim",
            "repair_if_required",
            "post_repair_synthesis",
            "post_repair_cosim",
            "promote_verified_baseline",
            "ppa_optimisation",
            "select_best_fully_verified_design",
        ],
        "shared_budget": {
            "max_iterations": 8,
            "max_model_calls": 8,
            "max_csim_calls": 10,
            "max_synthesis_calls": 10,
            "max_cosim_calls": 10,
            "max_total_tokens": 32768,
        },
    }
    index_path = output_root / "index.json"
    write_json(index_path, index)

    print(f"Created {len(generated)} repair-to-optimisation tasks:")
    for name in generated:
        print("-", name)
    print("Index:", index_path.relative_to(REPO_ROOT))
    print(
        "Recommended overnight plan: "
        f"{len(generated)} cases x 3 repetitions = {len(generated) * 3} full-agent runs"
    )


if __name__ == "__main__":
    main()
