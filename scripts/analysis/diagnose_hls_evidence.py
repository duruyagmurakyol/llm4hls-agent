#!/usr/bin/env python3

"""Diagnose generic HLS evidence and write a structured optimisation plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.analysis.hls_bottleneck_analyzer import analyse  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", help="JSON file containing generic HLS evidence")
    parser.add_argument("--output", help="Optional diagnosis JSON path")
    args = parser.parse_args()

    evidence_path = Path(args.evidence)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    diagnosis = analyse(evidence)

    output_path = Path(args.output) if args.output else evidence_path.with_name(
        evidence_path.stem + "_diagnosis.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8")

    primary = diagnosis["primary_diagnosis"]
    print("HLS bottleneck diagnosis")
    print(f"Category: {primary['category']}")
    print(f"Target: {primary['target']}")
    print(f"Confidence: {primary['confidence']:.2f}")
    print("Evidence:")
    for item in primary["evidence"]:
        print(f"  - {item}")
    print("Recommended transformations:")
    for item in primary["recommended_transformations"]:
        print(f"  - {item}")
    if primary["forbidden_transformations"]:
        print("Forbidden transformations:")
        for item in primary["forbidden_transformations"]:
            print(f"  - {item}")
    print(f"Stop recommended: {diagnosis['stop_recommended']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
