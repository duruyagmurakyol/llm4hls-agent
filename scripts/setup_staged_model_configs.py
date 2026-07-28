#!/usr/bin/env python3

"""Clone verified staged-repair configs for another SiliconFlow model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ("bicg", "atax")
CASES = (
    "staged_compile_then_functional",
    "staged_interface_then_functional",
    "staged_compile_compile_functional",
)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError("Model slug is empty")
    return slug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Exact SiliconFlow model identifier")
    parser.add_argument("--slug", default=None, help="Short identifier used in paths/results")
    args = parser.parse_args()

    model_slug = args.slug or slugify(args.model.split("/")[-1])
    created = 0

    for benchmark in BENCHMARKS:
        source_root = ROOT / f"configs/{benchmark}_iterative_qwen35"
        target_root = ROOT / f"configs/{benchmark}_iterative_{model_slug}"
        target_root.mkdir(parents=True, exist_ok=True)

        for case in CASES:
            source = source_root / f"{case}.json"
            if not source.is_file():
                raise SystemExit(f"Missing verified source config: {source}")

            config = json.loads(source.read_text(encoding="utf-8"))
            config["model"] = args.model
            config["experiment_id"] = f"{benchmark}_iterative_{model_slug}_{case}"

            target = target_root / f"{case}.json"
            target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {target.relative_to(ROOT)}")
            created += 1

    manifest = {
        "schema_version": 1,
        "model": args.model,
        "model_slug": model_slug,
        "benchmarks": list(BENCHMARKS),
        "cases": list(CASES),
        "configs_created": created,
    }
    manifest_path = ROOT / f"configs/staged_model_{model_slug}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
